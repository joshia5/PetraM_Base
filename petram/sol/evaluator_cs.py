import sys
import time
import numpy as np
import parser
import weakref
import traceback
import subprocess as sp
import cPickle
import binascii
try:
   import bz2  as bzlib
except ImportError:
   import zlib as bzlib   
from weakref import WeakKeyDictionary as WKD
from weakref import WeakValueDictionary as WVD

from petram.mfem_config import use_parallel
if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem

import multiprocessing as mp
from petram.sol.evaluators import Evaluator, EvaluatorCommon
from petram.sol.evaluator_mp import EvaluatorMPChild, EvaluatorMP

import thread
from threading import Timer, Thread
try:
    from Queue import Queue, Empty
except ImportError:
    from queue import Queue, Empty  # python 3.x
    
ON_POSIX = 'posix' in sys.builtin_module_names

wait_time = 0.3

def enqueue_output(p, queue, prompt):
    while True:
        line = p.stdout.readline()
        if len(line) == 0:
            time.sleep(wait_time)
            continue
        if line ==  (prompt + '\n'): break
        queue.put(line)
        if p.poll() is not None: return
    queue.put("??????")
    
def enqueue_output2(p, queue, prompt):
    # this assumes recievein two data (size and data)
    while True:
        line = p.stdout.readline()
        if len(line) == 0:
            time.sleep(wait_time)
            continue
        else:
            print("line", line)
            if line.startswith('z'):
                use_zlib = True
                size = int(line[1:])
            else:
                use_zlib = False                
                size = int(line)
            break
    line2 = p.stdout.read(size+1)
    line2 = binascii.a2b_hex(line2[:-1])
    if use_zlib:
        line2 = bzlib.decompress(line2)
    queue.put(line2)
    while True:
        line = p.stdout.readline()
        if len(line) == 0:
            time.sleep(wait_time)
            continue
        else:
            break
    if line !=  prompt + '\n':
         assert False, "I don't get prompt!??: " + line
    queue.put("??????")

def run_and_wait_for_prompt(p, prompt, verbose=True, withsize=False):    
    q = Queue()
    if withsize:
        t = Thread(target=enqueue_output2, args=(p, q, prompt))
    else:
        t = Thread(target=enqueue_output, args=(p, q, prompt))
    t.daemon = True # thread dies with the program
    t.start()

    lines = []; lastline = ""
    alive = True
    while lastline != "??????":
        try:  line = q.get_nowait() # or q.get(timeout=.1)
        except Empty:
            time.sleep(wait_time)                
            #print('no output yet' + str(p.poll()))
        else: # got line
            lines.append(line)
            lastline = lines[-1]            
        if p.poll() is not None:
            alive = False
            print('proces terminated')
            break

    if verbose:
        print("Data recieved: " + str(lines))
    else:
        print("Data length recieved: " + str([len(x) for x in lines]))
    return lines[:-1], alive

def run_with_timeout(timeout, default, f, *args, **kwargs):
    if not timeout:
        return f(*args, **kwargs)
    try:
        timeout_timer = Timer(timeout, thread.interrupt_main)
        timeout_timer.start()
        result = f(*args, **kwargs)
        return result
    except KeyboardInterrupt:
        return default
    finally:
        timeout_timer.cancel()
        
def wait_for_prompt(p, prompt = '?', verbose = True, withsize=False):
    return run_and_wait_for_prompt(p, prompt,
                                   verbose=verbose,
                                   withsize=withsize)
        
def start_connection(host = 'localhost', num_proc = 2, user = '', soldir = ''):
    if user != '': user = user+'@'
    p= sp.Popen("ssh " + user + host + " 'printf $PetraM'", shell=True,
                stdout=sp.PIPE)
    ans = p.stdout.readlines()[0].strip()
    command = ans+'/bin/evalsvr'
    if soldir != '':
        command = 'cd ' + soldir + ';' + command
    print command
    p = sp.Popen(['ssh', user + host, command], stdin = sp.PIPE,
                 stdout=sp.PIPE, stderr=sp.STDOUT,
                 close_fds = ON_POSIX,
                 universal_newlines = True)

    data, alive = wait_for_prompt(p, prompt = 'num_proc?')
    p.stdin.write(str(num_proc)+'\n')
    out, alive = wait_for_prompt(p)
    return p

def connection_test(host = 'localhost'):
    '''
    note that the data after process is terminated may be lost.
    '''
    p = start_connection(host = host, num_proc = 2)
    for i in range(5):
       p.stdin.write('test'+str(i)+'\n')
       out, alive = wait_for_prompt(p)
    p.stdin.write('e\n')
    out, alive = wait_for_prompt(p)

from petram.sol.evaluator_mp import EvaluatorMP
class EvaluatorServer(EvaluatorMP):
    def __init__(self, nproc = 2, logfile = 'queue'):
        return EvaluatorMP.__init__(self, nproc = nproc,
                                    logfile = logfile)
    
    def set_model(self, soldir):
        import os
        soldir = os.path.expanduser(soldir)        
        model_path = os.path.join(soldir, 'model.pmfm')
        if not os.path.exists(model_path):
           if 'case' in os.path.split(soldir)[-1]:
               model_path = os.path.join(os.path.dirname(soldir), 'model.pmfm')
        if not os.path.exists(model_path):
            assert False, "Model File not found: " + model_path
            
        self.tasks.put((3, model_path), join = True)

    
class EvaluatorClient(Evaluator):
    def __init__(self, nproc = 2, host = 'localhost',
                       soldir = '', user = ''):
        self.init_done = False        
        self.soldir = soldir
        self.solfiles = None
        self.nproc = nproc
        self.p = start_connection(host =  host,
                                  num_proc = nproc,
                                  user = user,
                                  soldir = soldir)
        self.failed = False

    def __del__(self):
        self.terminate_all()
        self.p = None

    def __call_server0(self, name, *params, **kparams):
        if self.p is None: return
        verbose = kparams.pop("verbose", False)
        
        command = [name, params, kparams]
        data = binascii.b2a_hex(cPickle.dumps(command))
        print("Sending request", command)
        self.p.stdin.write(data + '\n')
        
        output, alive = wait_for_prompt(self.p,
                                        verbose = verbose,
                                        withsize = True)
        if not alive:
           self.p = None
           return
        response = output[-1]
        try:
            result = cPickle.loads(response)
            if verbose:
                print("result", result)
        except:
            traceback.print_exc()
            print("response", response)
            print("output", output)
            assert False, "Unpickle failed"
        #print 'output is', result
        if result[0] == 'ok':
            return result[1]
        elif result[0] == 'echo':
            print(result[1])
        else:
            print(output, result)
            #assert False, result[1]
            message = ''.join([o[:75] for o in output])
            assert False, message

    def __call_server(self, name, *params, **kparams):        
        try:
            return self.__call_server0(name, *params, **kparams)
        except IOError:
            self.failed = True
            raise
        except:
            raise


    def set_model(self,  *params, **kparams):
        return self.__call_server('set_model', self.soldir)
        
    def set_solfiles(self,  *params, **kparams):
        kparams["verbose"] = True        
        return self.__call_server('set_solfiles', *params, **kparams)
        
    def make_agents(self,  *params, **kparams):
        return self.__call_server('make_agents', *params, **kparams)        
        
    def load_solfiles(self,  *params, **kparams):
        return self.__call_server('load_solfiles', *params, **kparams)        

    def set_phys_path(self,  *params, **kparams):
        return self.__call_server('set_phys_path', *params, **kparams)        
        
    def validate_evaluator(self,  *params, **kparams):
        if self.p is None: return False
        kparams["verbose"] = True
        return self.__call_server('validate_evaluator', *params, **kparams)

    def eval(self,  *params, **kparams):
        return self.__call_server('eval', *params, **kparams)
    
    def eval_probe(self,  *params, **kparams):
        return self.__call_server('eval_probe', *params, **kparams)

    def terminate_all(self):
        return self.__call_server('terminate_all')
        
    

