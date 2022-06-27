#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#define ND 3
#define LSTR(x) #x
#define STR(x) LSTR(x)
typedef struct {
  // primary shape
  npy_intp dims[ND];
  // derived quantities useful for index manipulation
  npy_intp strides[ND];
  npy_intp blocks[ND];
  npy_intp vol;
} latt_shape;

typedef struct {
  npy_intp* data;
  int len;
} silly_queue;
void init_queue(int max_len, silly_queue* queue) {
  queue->data = (npy_intp*)malloc(max_len * sizeof(npy_intp));
  queue->len = 0;
}
void deinit_queue(silly_queue* queue) {
  free(queue->data);
  queue->data = NULL;
  queue->len = 0;
}
npy_intp pop_queue(silly_queue* queue) {
  assert(queue->len > 0);
  return queue->data[--queue->len];
}
void push_queue(npy_intp elt, silly_queue* queue) {
  queue->data[queue->len++] = elt;
}

static npy_intp compute_comp(npy_intp idx, npy_intp ax, const latt_shape* shape) {
  return (idx % shape->blocks[ax]) / shape->strides[ax];
}

static npy_intp shift_site_idx(npy_intp idx, npy_intp diff, npy_intp ax, const latt_shape* shape) {
  if (diff < 0) {
    diff += shape->dims[ax];
  }
  npy_intp full_block_idx = idx - (idx % shape->blocks[ax]);
  npy_intp new_idx = ((idx + diff*shape->strides[ax]) % shape->blocks[ax]) + full_block_idx;
  for (int i = 0; i < ND; ++i) {
    if (i == ax) {
      assert(compute_comp(new_idx, i, shape) == (compute_comp(idx, i, shape) + diff) % shape->dims[i]);
    }
    else {
      assert(compute_comp(new_idx, i, shape) == compute_comp(idx, i, shape));
    }
  }
  return new_idx;
}

static npy_intp get_bond_idx(npy_intp site_idx, npy_intp mu, const latt_shape* shape) {
  return site_idx + mu*shape->vol;
}

static void assign_flip_clusters(const int* bonds, int* flip_mask, int* labels, const int* rand_bits, const latt_shape* shape) {
  int cur_label = 0;
  npy_intp rand_idx = 0;
  silly_queue queue;
  init_queue(shape->vol, &queue);
  for (npy_intp x = 0; x < shape->vol; ++x) {
    if (labels[x] != 0) continue;
    int cur_flip = rand_bits[rand_idx++];
    labels[x] = ++cur_label;
    flip_mask[x] = cur_flip;
    push_queue(x, &queue);
    while (queue.len > 0) {
      npy_intp y = pop_queue(&queue);
      for (int i = 0; i < ND; ++i) {
        npy_intp y_fwd = shift_site_idx(y, 1, i, shape);
        npy_intp y_bwd = shift_site_idx(y, -1, i, shape);
        if (bonds[get_bond_idx(y, i, shape)] && labels[y_fwd] != cur_label) {
          assert(labels[y_fwd] == 0);
          labels[y_fwd] = cur_label;
          flip_mask[y_fwd] = cur_flip;
          push_queue(y_fwd, &queue);
        }
        if (bonds[get_bond_idx(y_bwd, i, shape)] && labels[y_bwd] != cur_label) {
          assert(labels[y_bwd] == 0);
          labels[y_bwd] = cur_label;
          flip_mask[y_bwd] = cur_flip;
          push_queue(y_bwd, &queue);
        }
      }
    }
  }
  deinit_queue(&queue);

  // CHECK:
  /*
  for (npy_intp x = 0; x < shape->vol; ++x) {
    for (int i = 0; i < ND; ++i) {
      if (bonds[get_bond_idx(x, i, shape)]) {
        npy_intp x_fwd = shift_site_idx(x, 1, i, shape);
        assert(labels[x] == labels[x_fwd]);
      }
    }
  }
  */
}


static PyObject* sample_flip_mask(PyObject* self, PyObject* args) {
  PyArrayObject* bonds;
  PyArrayObject* rand_bits;

  if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &bonds, &PyArray_Type, &rand_bits))
    return NULL;

  // check `bonds`
  if (PyArray_NDIM(bonds) != (ND+1) || PyArray_DIM(bonds, 0) != ND || PyArray_TYPE(bonds) != NPY_INT) {
    PyErr_SetString(PyExc_RuntimeError, "bonds must be " STR(ND) "-dimensional integer link array");
    PyErr_Occurred();
    return NULL;
  }
  bonds = PyArray_GETCONTIGUOUS(bonds); // +1 ref

  // lattice shape
  latt_shape shape = {.vol = 1};
  npy_intp* bond_dims = PyArray_DIMS(bonds);
  for (int i = ND-1; i >= 0; --i) {
    shape.dims[i] = bond_dims[i+1];
    shape.strides[i] = shape.vol;
    shape.vol *= shape.dims[i];
    shape.blocks[i] = shape.vol;
  }
  // printf("Dims %ld %ld %ld\n", shape.dims[0], shape.dims[1], shape.dims[2]);
  // printf("Strides %ld %ld %ld\n", shape.strides[0], shape.strides[1], shape.strides[2]);
  // printf("Blocks %ld %ld %ld\n", shape.blocks[0], shape.blocks[1], shape.blocks[2]);
  // printf("Vol %ld\n", shape.vol);

  // check `rand_bits`
  if (PyArray_NDIM(rand_bits) != 1 || PyArray_DIM(rand_bits, 0) < shape.vol || PyArray_TYPE(rand_bits) != NPY_INT) {
    PyErr_SetString(PyExc_RuntimeError, "rand_bits must be a 1-dimensional list of at least V integers {0,1}");
    PyErr_Occurred();
    return NULL;
  }
  rand_bits = PyArray_GETCONTIGUOUS(rand_bits); // +1 ref

  // check shape and extract 
  if (ND*shape.vol != PyArray_SIZE(bonds)) {
    PyErr_SetString(PyExc_RuntimeError, "volume did not match bonds array for some reason");
    PyErr_Occurred();
    return NULL;
  }

  const int* rand_bits_ptr = (const int*)PyArray_DATA(rand_bits);
  const int* bond_ptr = (const int*)PyArray_DATA(bonds);

  // build output arrays:
  //  -- the flip mask with {0,1} per site
  //  -- the cluster labels with 1 ... n_cluster per site
  PyArrayObject* flip_mask = (PyArrayObject*) PyArray_ZEROS(
      3, shape.dims, NPY_INT, 0 /* C order */);
  PyArrayObject* labels = (PyArrayObject*) PyArray_ZEROS(
      3, shape.dims, NPY_INT, 0 /* C order */);
  int* flip_ptr = (int*)PyArray_DATA(flip_mask);
  int* labels_ptr = (int*)PyArray_DATA(labels);

  assign_flip_clusters(bond_ptr, flip_ptr, labels_ptr, rand_bits_ptr, &shape);

  Py_DECREF(bonds);
  Py_DECREF(rand_bits);

  return Py_BuildValue("(NN)", flip_mask, labels);
}


static PyMethodDef methods[] = {
  {"sample_flip_mask", sample_flip_mask, METH_VARARGS,
   "build clusters and sample flips"},
  {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION < 3
#error "Only Python3 supported"
#endif

static struct PyModuleDef module_defn = {
  PyModuleDef_HEAD_INIT,
  "cluster", "Fast evaluation of clusters",
  -1, methods
};

PyMODINIT_FUNC PyInit_cluster() {
  PyObject* module = PyModule_Create(&module_defn);
  if (module == NULL)
    return NULL;
  import_array();
  if (PyErr_Occurred())
    return NULL;
  return module;
}
