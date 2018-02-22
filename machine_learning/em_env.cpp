#define _GATAN_USE_STL_STRING		// Provide conversion from 'DM::String' to 'std::string'

#define _GATANPLUGIN_USES_LIBRARY_VERSION 2
#include "DMPlugInBasic.h"

#define _GATANPLUGIN_USE_CLASS_PLUGINMAIN
#include "DMPlugInMain.h"

#include "Python.h"
#include <numpy/arrayobject.h>

/* Get image from camera */
static PyObject* 
get_img(PyObject* self, PyObject* args)
{
	//Image dimensions
	int w, h;
	PyArg_ParseTuple(args, "i", &w, &h);

	//Get the image
	static Gatan::DM::Function __sFunction = (DM_FunctionToken)NULL;
	static const char *__sSignature = "BasicImage someFunction( )";
	Gatan::PlugIn::DM_Variant params[1];
	GatanPlugIn::gDigitalMicrographInterface.CallFunction(__sFunction.get_ptr(), 1, params, __sSignature);

	//Get the image
	Gatan::float32 *data = params[0].v_float32_ref;

	//Wrap for numpy
	npy_intp dims[2] = { w, h };
	return PyArray_SimpleNewFromData(2, dims, PyArray_FLOAT, data);
}

static PyMethodDef methods[] = {
	{ "get_img", get_img, METH_VARARGS, "Get image from camera" },
	{ NULL, NULL, 0, NULL }
};

static struct PyModuleDef em_env_module =
{
	PyModuleDef_HEAD_INIT,
	"em_env", "Methods to interact with the electron microscope",
	-1,
	methods
};


PyMODINIT_FUNC
PyInit_em_env(void)
{
	return PyModule_Create(&em_env_module);
}