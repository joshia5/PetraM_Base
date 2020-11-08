from petram.solver.std_solver_model import StdSolver, StandardSolver
import pyCore
import mfem.par as mfem
from mfem.par import intArray
from mfem.par import Vector
from mfem.par import DenseMatrix
from mfem._par.pumi import ParPumiMesh
from mfem._par.pumi import ParMesh2ParPumiMesh
import os
import numpy as np
import math

from petram.model import Model
from petram.solver.solver_model import Solver
import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('StdMeshAdaptSolver')
rprint = debug.regular_print('StdMeshAdaptSolver')


def get_field_z_averaged(pumi_mesh, field_name, field_type, grid):
  # get the mfem mesh and fespace
  fes = grid.ParFESpace()
  mesh = fes.GetParMesh()

def cap_field_by_size(pumi_mesh,
                      field,
                      max_val):

  dim = pumi_mesh.getDimension()
  fs = pyCore.getShape(field)
  for d in range(dim+1):
    if not fs.hasNodesIn(d):
      continue
    it = pumi_mesh.begin(d)
    while True:
      ent = pumi_mesh.iterate(it)
      if not ent:
        break
      e_type = pumi_mesh.getType(ent)
      non = fs.countNodesOn(e_type)
      for n in range(non):
        vec = pyCore.Vector3()
        pyCore.getVector(field, ent, n, vec)
        npvec = vector3_to_nparray(vec)
        s = math.sqrt(np.dot(npvec, npvec))
        if s > max_val:
          npvec *= (max_val/s)
        pyCore.setVector(field, ent, n,
            pyCore.Vector3(npvec[0], npvec[1], npvec[2]))

    pumi_mesh.end(it)

def process_size_field(pumi_mesh, size_field, rmin = 0.25, rmax = 1.5):
  current_size = pumi_mesh.getCurrentIsoSize("current_size")
  fs = pyCore.getShape(current_size)

  # min_size0 = pumi_mesh.getMinOfScalarField(size_field);
  # max_size0 = pumi_mesh.getMaxOfScalarField(size_field);
  # print("before process min and max are ", min_size0, max_size0)

  # min_current = pumi_mesh.getMinOfScalarField(current_size);
  # max_current = pumi_mesh.getMaxOfScalarField(current_size);
  # print("before process min and max of current are ", min_current, max_current)



  # compute the relative size
  relative_size = pyCore.createField(pumi_mesh, "relative_size", pyCore.SCALAR, fs)
  it = pumi_mesh.begin(0)
  while True:
    ent = pumi_mesh.iterate(it)
    if not ent:
      break
    if not pumi_mesh.isOwned(ent):
      continue
    c = pyCore.getScalar(current_size, ent, 0)
    s = pyCore.getScalar(size_field, ent, 0)
    pyCore.setScalar(relative_size, ent, 0, s/c)
  pumi_mesh.end(it)
  pyCore.synchronize(relative_size)

  # find the min relative_size across the whole mesh
  min_size = pumi_mesh.getMinOfScalarField(relative_size);
  # max_size = pumi_mesh.getMaxOfScalarField(relative_size);
  # print("before modify  min and max of relative are ", min_size, max_size)
  # scale the relative_size such that the min is equal to rmin
  # and cap it by rmax
  it = pumi_mesh.begin(0)
  while True:
    ent = pumi_mesh.iterate(it)
    if not ent:
      break
    if not pumi_mesh.isOwned(ent):
      continue
    val = pyCore.getScalar(relative_size, ent, 0)
    val = val * rmin / min_size
    if val > rmax:
      val = rmax
    pyCore.setScalar(relative_size, ent, 0, val)
  pumi_mesh.end(it)
  pyCore.synchronize(relative_size)

  # min_size_ = pumi_mesh.getMinOfScalarField(relative_size);
  # max_size_ = pumi_mesh.getMaxOfScalarField(relative_size);
  # print("after modify  min and max of relative are ", min_size_, max_size_)

  # update the size_field to be relative_size*current_size
  it = pumi_mesh.begin(0)
  while True:
    ent = pumi_mesh.iterate(it)
    if not ent:
      break
    if not pumi_mesh.isOwned(ent):
      continue
    r = pyCore.getScalar(relative_size, ent, 0)
    cs = pyCore.getScalar(current_size, ent, 0)
    pyCore.setScalar(size_field, ent, 0, r*cs)
  pumi_mesh.end(it)
  pyCore.synchronize(size_field)

  # min_size2 = pumi_mesh.getMinOfScalarField(size_field);
  # max_size2 = pumi_mesh.getMaxOfScalarField(size_field);
  # print("after process min and max are ", min_size2, max_size2)


  # clean up relative_size current_size
  pumi_mesh.removeField(relative_size)
  pumi_mesh.removeField(current_size)
  pyCore.destroyField(relative_size)
  pyCore.destroyField(current_size)
  return min_size


def project_np_array_onto_plane(e, normal):
  e_dot_normal = np.dot(e, normal)
  return e - e_dot_normal * normal

def limit_refine_level(pumi_mesh, sizefield, level):
  # TODO: this needs to be updated for parallel runs to use cavity ops
  it = pumi_mesh.begin(0)
  while True:
    ent = pumi_mesh.iterate(it)
    if not ent:
      break
    current_size = pumi_mesh.measureSize(ent)
    computed_size = pyCore.getScalar(sizefield, ent, 0)
    if computed_size < current_size / (2**level):
      computed_size = current_size / (2**level)
    # if computed_size > current_size:
    #   computed_size = current_size;
    pyCore.setScalar(sizefield, ent, 0, computed_size)
  pumi_mesh.end(it)
  pyCore.synchronize(sizefield)

def limit_coarsen(pumi_mesh, sizefield, ratio):
  # TODO: this needs to be updated for parallel runs to use cavity ops
  it = pumi_mesh.begin(0)
  while True:
    ent = pumi_mesh.iterate(it)
    if not ent:
      break
    pyCore.setScalar(count_field, ent, 0, 0.0)
    p = pyCore.Vector3(0.0, 0.0, 0.0)
    pyCore.setVector(sol_field, ent, 0, p)
  pumi_mesh.end(it)
  pyCore.synchronize(sizefield)

  it = pumi_mesh.begin(dim)
  eid = 0
  while True:
    ent = pumi_mesh.iterate(it)
    if not ent:
      break

    felem = fes.GetFE(eid)
    elem_vert = mfem.geom.Geometry().GetVertices(felem.GetGeomType())

    vval = DenseMatrix()
    pmat = DenseMatrix()
    grid.GetVectorValues(eid, elem_vert, vval, pmat)
    pyCore.PCU_ALWAYS_ASSERT(vval.Width() == 4)
    pyCore.PCU_ALWAYS_ASSERT(vval.Height() == 3)

    mfem_vids = mesh.GetElementVertices(eid)

    for i in range(4):
      current_count = pumi_mesh.getVertScalarField(count_field, ent, i, 0)
      current_sol = pumi_mesh.getVertVectorField(sol_field, ent, i, 0)
      pumi_vid = pumi_mesh.getVertNumbering(numbering, ent, i, 0, 0)
      # this is the index of pumi_vid in mfem_vids list
      # (note that we need to do this because of ReorientTet call)
      j = mfem_vids.index(pumi_vid)
      pumi_mesh.setVertScalarField(count_field, ent, i, 0, current_count + 1.0)
      # pumi_mesh.setVertVectorField(sol_field, ent, i, 0, current_sol.x()+vval[0,j],
      #                                                    current_sol.y()+vval[1,j],
      #                                                    current_sol.z()+vval[2,j])
      pumi_mesh.setVertVectorField(sol_field, ent, i, 0, vval[0,j],
                                                         vval[1,j],
                                                         vval[2,j])

    eid = eid + 1

  pumi_mesh.end(it)

  # do the average since each vertex gets a value from multiple tets
  it = pumi_mesh.begin(0)
  while True:
    ent = pumi_mesh.iterate(it)
    if not ent:
      break
    current_count = pyCore.getScalar(count_field, ent, 0)
    current_sol = pyCore.Vector3()
    pyCore.getVector(sol_field, ent, 0, current_sol)
    # avg_sol = pyCore.Vector3(current_sol.x() / current_count,
    #                          current_sol.y() / current_count,
    #                          current_sol.z() / current_count)
    sol_limit = 5000.0
    avg_sol_x = sol_limit
    avg_sol_y = sol_limit
    avg_sol_z = sol_limit
    if abs(current_sol.x()) < sol_limit:
      avg_sol_x = current_sol.x()
    if abs(current_sol.y()) < sol_limit:
      avg_sol_y = current_sol.y()
    if abs(current_sol.z()) < sol_limit:
      avg_sol_z = current_sol.z()
    avg_sol = pyCore.Vector3(avg_sol_x,
                             avg_sol_y,
                             avg_sol_z)
    # mag = sqrt(avg_sol.x() * avg_sol.x() +
    #          avg_sol.y() * avg_sol.y() +
    #          avg_sol.z() * avg_sol.z())
    mag = math.sqrt(avg_sol.x() * avg_sol.x()+
    	            avg_sol.y() * avg_sol.y()+
    	            avg_sol.z() * avg_sol.z())
    pyCore.setScalar(field_z, ent, 0, mag)
    pyCore.setVector(sol_field, ent, 0, avg_sol)



  pumi_mesh.end(it)
  pumi_mesh.removeField(count_field)
  pyCore.destroyField(count_field)
  # pumi_mesh.removeField(sol_field)
  # pyCore.destroyField(sol_field)
  return field_z


def get_curl_ip_field(pumi_mesh, field_name, field_type, field_order, grid):
  # get the mfem mesh and fespace
  fes = grid.ParFESpace()
  mesh = fes.GetParMesh()


  # create the necessary fields
  curl_field = pyCore.createIPField(pumi_mesh, field_name, field_type, field_order)
  # curl_field_shape = pyCore.getShape(curl_field)

  dim = pumi_mesh.getDimension()

  it = pumi_mesh.begin(dim)
  eid = 0
  while True:
    ent = pumi_mesh.iterate(it)
    if not ent:
      break

    elem_transformation = fes.GetElementTransformation(eid)
    curl_vec = mfem.Vector()
    grid.GetCurl(elem_transformation, curl_vec)

    # for i in range(4):
    #   current_count = pumi_mesh.getVertScalarField(count_field, ent, i, 0)
    #   current_sol = pumi_mesh.getVertVectorField(sol_field, ent, i, 0)
    #   pumi_vid = pumi_mesh.getVertNumbering(numbering, ent, i, 0, 0)
    #   # this is the index of pumi_vid in mfem_vids list
    #   # (note that we need to do this because of ReorientTet call)
    #   j = mfem_vids.index(pumi_vid)
    #   pumi_mesh.setVertScalarField(count_field, ent, i, 0, current_count + 1.0)
    #   pumi_mesh.setVertVectorField(sol_field, ent, i, 0, current_sol.x()+vval[0,j],
    #                                                      current_sol.y()+vval[1,j],
    #                                                      current_sol.z()+vval[2,j])
    # pyCore.setComponents(curl_field, ent, 0, curl_vec.GetData())
    pumi_mesh.setIPxyz(curl_field, ent, 0, curl_vec[0], curl_vec[1], curl_vec[2])

    eid = eid + 1

  pumi_mesh.end(it)

  return curl_field

def limit_refine_level(pumi_mesh, sizefield, level):
  it = pumi_mesh.begin(0)
  while True:
    ent = pumi_mesh.iterate(it)
    if not ent:
      break
    current_size = pumi_mesh.measureSize(ent)
    computed_size = pyCore.getScalar(sizefield, ent, 0)
    if computed_size < current_size / (2**level):
      computed_size = current_size / (2**level)
    if computed_size > current_size:
      computed_size = current_size;
    pyCore.setScalar(sizefield, ent, 0, computed_size)
  pumi_mesh.end(it)


class StdMeshAdaptSolver(StdSolver):
    @classmethod
    def fancy_menu_name(self):
        return 'AMR Stationary'

    @classmethod
    def fancy_tree_name(self):
        return 'AMR Stationary'

    def panel1_param(self):
        return [  # ["Initial value setting",   self.init_setting,  0, {},],
            ["physics model",   self.phys_model,  0, {}, ],
            ["initialize solution only", self.init_only,  3, {"text": ""}],
            [None,
             self.clear_wdir,  3, {"text": "clear working directory"}],
            [None,
             self.assemble_real,  3, {"text": "convert to real matrix (complex prob.)"}],
            [None,
             self.save_parmesh,  3, {"text": "save parallel mesh"}],
            [None,
             self.use_profiler,  3, {"text": "use profiler"}],
            ["indicator",   self.mesh_adapt_indicator,  0, {}],
            ["#mesh adapt",   self.mesh_adapt_num,  0, {}, ], ]

    def attribute_set(self, v):
        super(StdMeshAdaptSolver, self).attribute_set(v)
        v["mesh_adapt_indicator"] = ""
        v["mesh_adapt_num"] = 0
        return v

    def get_panel1_value(self):
        return (  # self.init_setting,
            self.phys_model,
            self.init_only,
            self.clear_wdir,
            self.assemble_real,
            self.save_parmesh,
            self.use_profiler,
            self.mesh_adapt_indicator,
            self.mesh_adapt_num)

    def import_panel1_value(self, v):
        #self.init_setting = str(v[0])
        self.phys_model = str(v[0])
        self.init_only = v[1]
        self.clear_wdir = v[2]
        self.assemble_real = v[3]
        self.save_parmesh = v[4]
        self.use_profiler = v[5]
        self.mesh_adapt_indicator = v[6]
        self.mesh_adapt_num = int(v[7])

    @debug.use_profiler
    def run(self, engine, is_first=True, return_instance=False):
        dprint1("Entering run", is_first, self.fullpath())
        if self.clear_wdir:
            engine.remove_solfiles()

        instance = StandardMeshAdaptSolver(self, engine)
        instance.set_blk_mask()
        if return_instance:
            return instance
        # We dont use probe..(no need...)
        # instance.configure_probes(self.probe)

        if self.init_only:
            engine.sol = engine.assembled_blocks[1][0]
            instance.sol = engine.sol
        else:
            if is_first:
                instance.assemble()
                is_first = False
            instance.solve()

        adapt_loop_no = 0;
        # save the initial (before adapt) solution
        instance.ma_save_mfem(adapt_loop_no, parmesh = True)
        while adapt_loop_no < int(self.mesh_adapt_num):
            engine.sol = instance.sol
            dprint1(debug.format_memory_usage())


            x = engine.r_x[0]
            y = engine.i_x[0]
            # par_pumi_mesh = self.root()._par_pumi_mesh

            par_pumi_mesh = ParMesh2ParPumiMesh(engine.meshes[0])
            # par_pumi_emesh = ParMesh2ParPumiMesh(engine.emeshes[0])
            pumi_mesh = self.root()._pumi_mesh

            # transfer the e field to a nedelec field in pumi
            e_real = pyCore.createField(pumi_mesh,
                                        "e_real_nd",
                                        pyCore.SCALAR,
                                        pyCore.getNedelec(order))
            e_imag = pyCore.createField(pumi_mesh,
                                        "e_imag_nd",
                                        pyCore.SCALAR,
                                        pyCore.getNedelec(order))

	    # nodal h1 fields for visualization
            e_real_projected = pyCore.createField(pumi_mesh,
                                                  "e_real_projected",
                                                   pyCore.VECTOR,
                                                   pyCore.getH1Shape(order))
            e_imag_projected = pyCore.createField(pumi_mesh,
                                                  "e_imag_projected",
                                                   pyCore.VECTOR,
                                                   pyCore.getH1Shape(order))

	    # integration points for error estimation
            ip_order = 2
            e_real_ip = pyCore.createIPField(pumi_mesh,
                                             "e_real_ip",
                                             pyCore.VECTOR,
                                             ip_order)
            e_imag_ip = pyCore.createIPField(pumi_mesh,
                                           "e_imag_ip",
                                           pyCore.VECTOR,
                                           ip_order)

            par_pumi_mesh.NedelecFieldMFEMtoPUMI(pumi_mesh, x, e_real)
            par_pumi_mesh.NedelecFieldMFEMtoPUMI(pumi_mesh, y, e_imag)

            pyCore.projectNedelecField(e_real_projected, e_real)
            pyCore.projectNedelecField(e_imag_projected, e_imag)

            pyCore.projectNedelecField(e_real_ip, e_real)
            pyCore.projectNedelecField(e_imag_ip, e_imag)

            cap_field_by_size(pumi_mesh, e_real_ip, 15.0);
            cap_field_by_size(pumi_mesh, e_imag_ip, 15.0);


            # pumi_projected_nedelec_field = pumi_projected_nedelec_field_real
            # print("user choose the indicator ", self.mesh_adapt_indicator)
            # indicator_field = 0;
            # if self.mesh_adapt_indicator == "E":
            #   indicator_field = pumi_projected_nedelec_field
            # elif self.mesh_adapt_indicator == "Emag":
            #   indicator_field = get_field_component(pumi_mesh, pumi_projected_nedelec_field, 0)
            # elif self.mesh_adapt_indicator == "Ex":
            #   indicator_field = get_field_component(pumi_mesh, pumi_projected_nedelec_field, 1)
            # elif self.mesh_adapt_indicator == "Ey":
            #   indicator_field = get_field_component(pumi_mesh, pumi_projected_nedelec_field, 2)
            # elif self.mesh_adapt_indicator == "Ez":
            #   indicator_field = get_field_component(pumi_mesh, pumi_projected_nedelec_field, 3)
            # else:
            #   assert False, "wrong indicator selected"

            # ip_field = pyCore.getGradIPField(amplitude_r, "mfem_grad_ip", 2)
            # size_field = pyCore.getTargetSPRSizeField(ip_field, int(pumi_mesh.count(3)*2) , 0.125, 2.)
            # limit_refine_level(pumi_mesh, size_field, 3)
            # limit_coarsen(pumi_mesh, size_field, 1.5)
            # relative_size_field = compute_relative_size(pumi_mesh, size_field)

            size_field = pyCore.getSPRSizeField(e_real_ip, float(self.mesh_adapt_ar))
            minsize = process_size_field(pumi_mesh, size_field, 0.25, 1.5)
            print("min size is ", minsize)

            min_of_size_field = pumi_mesh.getMinOfScalarField(size_field)

            mesh_shape = pumi_mesh.getShape()
            mesh_order = mesh_shape.getOrder()
            if mesh_order == 1:
                native_name = "pumi_mesh_before_adapt_"+str(adapt_loop_no)+"_.smb";
                pumi_mesh.writeNative(native_name)

            before_prefix = "before_adapt_"+str(adapt_loop_no);
            if mesh_order == 1:
                pyCore.writeASCIIVtkFiles(before_prefix, pumi_mesh);
            # else:
            #     pyCore.writeCurvedVtuFiles(pumi_mesh, 4, 8, before_prefix);
            #     pyCore.writeCurvedWireFrame(pumi_mesh, 8, before_prefix);

            pumi_mesh.removeField(e_real)
            pumi_mesh.removeField(e_imag)
            pyCore.destroyField(e_real)
            pyCore.destroyField(e_imag)

            pumi_mesh.removeField(e_real_ip)
            pyCore.destroyField(e_real_ip)
            pumi_mesh.removeField(e_imag_ip)
            pyCore.destroyField(e_imag_ip)

            adapt_input = pyCore.configure(pumi_mesh, size_field)
            adapt_input.shouldFixShape = True
            adapt_input.shouldCoarsen = True
            adapt_input.maximumIterations = 3
            adapt_input.goodQuality = 0.35 * 0.35 * 0.35 # mean-ratio cubed

            if mesh_order == 1:
                pyCore.adaptVerbose(adapt_input)
            else:
                pyCore.crvadapt(adapt_input)

            after_prefix = "after_adapt_"+str(adapt_loop_no);
            if mesh_order == 1:
                pyCore.writeASCIIVtkFiles(after_prefix, pumi_mesh);
            # else:
            #     pyCore.writeCurvedVtuFiles(pumi_mesh, 4, 8, after_prefix);
            #     pyCore.writeCurvedWireFrame(pumi_mesh, 8, after_prefix);

            if mesh_order == 1:
                native_name = "pumi_mesh_after_adapt_"+str(adapt_loop_no)+".smb";
                pumi_mesh.writeNative(native_name)

            # clean up rest of the fields
            pumi_mesh.removeField(e_real_projected)
            pyCore.destroyField(e_real_projected)
            # pumi_mesh.removeField(e_real_projected_clean)
            # pyCore.destroyField(e_real_projected_clean)

            pumi_mesh.removeField(e_imag_projected)
            pyCore.destroyField(e_imag_projected)
            # pumi_mesh.removeField(e_imag_projected_clean)
            # pyCore.destroyField(e_imag_projected_clean)

            pumi_mesh.removeField(size_field)
            pyCore.destroyField(size_field)

            # pumi_mesh.removeField(relative_size_field)
            # pyCore.destroyField(relative_size_field)

            # pumi_mesh.removeField(amplitude_r)

            adapted_mesh = mfem.ParMesh(pyCore.PCU_Get_Comm(), pumi_mesh)
            # add the boundary attributes
            dim = pumi_mesh.getDimension()
            it = pumi_mesh.begin(dim-1)
            bdr_cnt = 0
            while True:
                e = pumi_mesh.iterate(it)
                if not e: break
                model_tag  = pumi_mesh.getModelTag(pumi_mesh.toModel(e))
                model_type = pumi_mesh.getModelType(pumi_mesh.toModel(e))
                if model_type == (dim-1):
                    adapted_mesh.GetBdrElement(bdr_cnt).SetAttribute(model_tag)
                    bdr_cnt += 1
            pumi_mesh.end(it)
            it = pumi_mesh.begin(dim)
            elem_cnt = 0
            while True:
                e = pumi_mesh.iterate(it)
                if not e: break
                model_tag  = pumi_mesh.getModelTag(pumi_mesh.toModel(e))
                model_type = pumi_mesh.getModelType(pumi_mesh.toModel(e))
                if model_type == dim:
                    adapted_mesh.SetAttribute(elem_cnt, model_tag)
                    elem_cnt += 1
            pumi_mesh.end(it)
            adapted_mesh.SetAttributes()

            par_pumi_mesh.UpdateMesh(adapted_mesh)
            # update the _par_pumi_mesh and _pumi_mesh as well

            # self.root()._par_pumi_mesh = par_pumi_mesh
            # self.root()._pumi_mesh = pumi_mesh

            # engine.meshes[0] = mfem.ParMesh(pyCore.PCU_Get_Comm(), par_pumi_mesh)
            engine.emeshes[0] = engine.meshes[0]
            # reorient the new mesh
            engine.meshes[0].ReorientTetMesh()

            # the rest of the updates happen here
            instance.ma_update_form_sol_variables()
            instance.ma_init()
            instance.set_blk_mask()
            instance.ma_update_assemble()
            instance.solve()
            adapt_loop_no = adapt_loop_no + 1
            instance.ma_save_mfem(adapt_loop_no, parmesh = True)
        return is_first

class StandardMeshAdaptSolver(StandardSolver):
    def ma_update_assemble(self, inplace=True):
        engine = self.engine
        phys_target = self.get_phys()
        phys_range = self.get_phys_range()

        engine.run_verify_setting(phys_target, self.gui)
        engine.run_assemble_mat(phys_target, phys_range, update=False)
        engine.run_assemble_b(phys_target, update=False)
        self.engine.run_assemble_blocks(self.compute_A,self.compute_rhs,inplace=True,update=False)
        self.assembled = True
    def ma_update_form_sol_variables(self):
        engine = self.engine
        phys_target = self.get_phys()
        phys_range = self.get_phys_range()
        num_matrix = engine.n_matrix
        engine.set_formblocks(phys_target, phys_range, num_matrix)
        # mesh should be already updated
        # not all of run_alloc_sol needs to be called here
        # so call the ones that are necessary
        for phys in phys_range:
            engine.initialize_phys(phys, update=True)
        for j in range(engine.n_matrix):
            engine.accept_idx = j
            engine.r_x.set_no_allocator()
            engine.i_x.set_no_allocator()
    def ma_init(self):
        engine = self.engine
        phys_target = self.get_phys()
        phys_range = self.get_phys_range()
        engine.run_apply_essential(phys_target, phys_range, update=False)
        engine.run_fill_X_block(update=False)
    # Save the mfem mesh and solution at the beginning of each adapt loop.
    # Solutions will be saved in sub-folders named case_xxx, where xxx is adapt-solve loop #
    def ma_save_mfem(self, loop_num, parmesh = True):
        engine = self.engine
        phys_target = self.get_phys()

        # take care of sub-folders here
        sub_folder = "case_" + str(loop_num).zfill(3)
        od = os.getcwd()
        path = os.path.join(od, sub_folder)
        engine.mkdir(path)
        os.chdir(path)
        engine.cleancwd()

        # save the mesh and solution in new location
        sol, sol_extra = engine.split_sol_array(self.sol)
        engine.recover_sol(sol)
        extra_data = engine.process_extra(sol_extra)


        engine.save_sol_to_file(phys_target,
                            skip_mesh = False,
                            mesh_only = False,
                            save_parmesh = True)
        engine.save_extra_to_file(extra_data)

        # go back to the original working directory
        os.chdir(od)
