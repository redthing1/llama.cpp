import core.stdc.math;
import core.stdc.errno;
import core.stdc.stdio;
import core.stdc.stdlib;
import core.stdc.time;
import core.stdc.ctype;
import core.stdc.locale;
import core.stdc.stdarg;
import core.stdc.wchar_;
import core.stdc.stddef;
import core.stdc.stdint;

import core.stdc.config;
import std.bitmanip : bitfields;
import std.conv : emplace;

bool isModuleAvailable(alias T)() {
    mixin("import " ~ T ~ ";");
    static if (__traits(compiles, (mixin(T)).stringof))
        return true;
    else
        return false;
}
    
static if (__traits(compiles, isModuleAvailable!"nsgen" )) 
    static import nsgen;

struct CppClassSizeAttr
{
    alias size this;
    size_t size;
}
CppClassSizeAttr cppclasssize(size_t a) { return CppClassSizeAttr(a); }

struct CppSizeAttr
{
    alias size this;
    size_t size;
}
CppSizeAttr cppsize(size_t a) { return CppSizeAttr(a); }

struct CppMethodAttr{}
CppMethodAttr cppmethod() { return CppMethodAttr(); }

struct PyExtract{}
auto pyExtract(string name = null) { return PyExtract(); }

mixin template RvalueRef()
{
    alias T = typeof(this);
    static assert (is(T == struct));

    @nogc @safe
    ref const(T) byRef() const pure nothrow return
    {
        return this;
    }
}


enum GGML_MAX_DIMS = 4;
enum GGML_MAX_NODES = 4096;
enum GGML_MAX_PARAMS = 16;
enum GGML_MAX_CONTEXTS = 64;
enum GGML_MAX_OPT = 4;
alias ggml_fp16_t = uint16_t;

extern(C)
float ggml_fp16_to_fp32(ggml_fp16_t x);

extern(C)
ggml_fp16_t ggml_fp32_to_fp16(float x);

enum ggml_type
{
    GGML_TYPE_Q4_0 = 0, 
    GGML_TYPE_Q4_1 = 1, 
    GGML_TYPE_I8 = 2, 
    GGML_TYPE_I16 = 3, 
    GGML_TYPE_I32 = 4, 
    GGML_TYPE_F16 = 5, 
    GGML_TYPE_F32 = 6, 
    GGML_TYPE_COUNT = 7, 
}

alias GGML_TYPE_Q4_0 = ggml_type.GGML_TYPE_Q4_0;
alias GGML_TYPE_Q4_1 = ggml_type.GGML_TYPE_Q4_1;
alias GGML_TYPE_I8 = ggml_type.GGML_TYPE_I8;
alias GGML_TYPE_I16 = ggml_type.GGML_TYPE_I16;
alias GGML_TYPE_I32 = ggml_type.GGML_TYPE_I32;
alias GGML_TYPE_F16 = ggml_type.GGML_TYPE_F16;
alias GGML_TYPE_F32 = ggml_type.GGML_TYPE_F32;
alias GGML_TYPE_COUNT = ggml_type.GGML_TYPE_COUNT;

enum ggml_op
{
    GGML_OP_NONE = 0, 
    GGML_OP_DUP = 1, 
    GGML_OP_ADD = 2, 
    GGML_OP_SUB = 3, 
    GGML_OP_MUL = 4, 
    GGML_OP_DIV = 5, 
    GGML_OP_SQR = 6, 
    GGML_OP_SQRT = 7, 
    GGML_OP_SUM = 8, 
    GGML_OP_MEAN = 9, 
    GGML_OP_REPEAT = 10, 
    GGML_OP_ABS = 11, 
    GGML_OP_SGN = 12, 
    GGML_OP_NEG = 13, 
    GGML_OP_STEP = 14, 
    GGML_OP_RELU = 15, 
    GGML_OP_GELU = 16, 
    GGML_OP_SILU = 17, 
    GGML_OP_NORM = 18, 
    GGML_OP_RMS_NORM = 19, 
    GGML_OP_MUL_MAT = 20, 
    GGML_OP_SCALE = 21, 
    GGML_OP_CPY = 22, 
    GGML_OP_RESHAPE = 23, 
    GGML_OP_VIEW = 24, 
    GGML_OP_PERMUTE = 25, 
    GGML_OP_TRANSPOSE = 26, 
    GGML_OP_GET_ROWS = 27, 
    GGML_OP_DIAG_MASK_INF = 28, 
    GGML_OP_SOFT_MAX = 29, 
    GGML_OP_ROPE = 30, 
    GGML_OP_CONV_1D_1S = 31, 
    GGML_OP_CONV_1D_2S = 32, 
    GGML_OP_FLASH_ATTN = 33, 
    GGML_OP_FLASH_FF = 34, 
    GGML_OP_COUNT = 35, 
}

alias GGML_OP_NONE = ggml_op.GGML_OP_NONE;
alias GGML_OP_DUP = ggml_op.GGML_OP_DUP;
alias GGML_OP_ADD = ggml_op.GGML_OP_ADD;
alias GGML_OP_SUB = ggml_op.GGML_OP_SUB;
alias GGML_OP_MUL = ggml_op.GGML_OP_MUL;
alias GGML_OP_DIV = ggml_op.GGML_OP_DIV;
alias GGML_OP_SQR = ggml_op.GGML_OP_SQR;
alias GGML_OP_SQRT = ggml_op.GGML_OP_SQRT;
alias GGML_OP_SUM = ggml_op.GGML_OP_SUM;
alias GGML_OP_MEAN = ggml_op.GGML_OP_MEAN;
alias GGML_OP_REPEAT = ggml_op.GGML_OP_REPEAT;
alias GGML_OP_ABS = ggml_op.GGML_OP_ABS;
alias GGML_OP_SGN = ggml_op.GGML_OP_SGN;
alias GGML_OP_NEG = ggml_op.GGML_OP_NEG;
alias GGML_OP_STEP = ggml_op.GGML_OP_STEP;
alias GGML_OP_RELU = ggml_op.GGML_OP_RELU;
alias GGML_OP_GELU = ggml_op.GGML_OP_GELU;
alias GGML_OP_SILU = ggml_op.GGML_OP_SILU;
alias GGML_OP_NORM = ggml_op.GGML_OP_NORM;
alias GGML_OP_RMS_NORM = ggml_op.GGML_OP_RMS_NORM;
alias GGML_OP_MUL_MAT = ggml_op.GGML_OP_MUL_MAT;
alias GGML_OP_SCALE = ggml_op.GGML_OP_SCALE;
alias GGML_OP_CPY = ggml_op.GGML_OP_CPY;
alias GGML_OP_RESHAPE = ggml_op.GGML_OP_RESHAPE;
alias GGML_OP_VIEW = ggml_op.GGML_OP_VIEW;
alias GGML_OP_PERMUTE = ggml_op.GGML_OP_PERMUTE;
alias GGML_OP_TRANSPOSE = ggml_op.GGML_OP_TRANSPOSE;
alias GGML_OP_GET_ROWS = ggml_op.GGML_OP_GET_ROWS;
alias GGML_OP_DIAG_MASK_INF = ggml_op.GGML_OP_DIAG_MASK_INF;
alias GGML_OP_SOFT_MAX = ggml_op.GGML_OP_SOFT_MAX;
alias GGML_OP_ROPE = ggml_op.GGML_OP_ROPE;
alias GGML_OP_CONV_1D_1S = ggml_op.GGML_OP_CONV_1D_1S;
alias GGML_OP_CONV_1D_2S = ggml_op.GGML_OP_CONV_1D_2S;
alias GGML_OP_FLASH_ATTN = ggml_op.GGML_OP_FLASH_ATTN;
alias GGML_OP_FLASH_FF = ggml_op.GGML_OP_FLASH_FF;
alias GGML_OP_COUNT = ggml_op.GGML_OP_COUNT;

extern(C)
@cppclasssize(160) align(8)
struct ggml_tensor
{


    @cppsize(0) public ggml_type type;
    @cppsize(0) public int n_dims;
    @cppsize(0) public int[4] ne;
    @cppsize(0) public size_t[4] nb;
    @cppsize(0) public ggml_op op;
    @cppsize(0) public bool is_param;
    @cppsize(0) public ggml_tensor* grad;
    @cppsize(0) public ggml_tensor* src0;
    @cppsize(0) public ggml_tensor* src1;
    @cppsize(0) public ggml_tensor*[4] opt;
    @cppsize(0) public int n_tasks;
    @cppsize(0) public int perf_runs;
    @cppsize(0) public int64_t perf_cycles;
    @cppsize(0) public int64_t perf_time_us;
    @cppsize(0) public void* data;
    @cppsize(0) public char[8] padding;
}
extern(C)
@cppclasssize(98360) align(8)
struct ggml_cgraph
{


    @cppsize(0) public int n_nodes;
    @cppsize(0) public int n_leafs;
    @cppsize(0) public int n_threads;
    @cppsize(0) public size_t work_size;
    @cppsize(0) public ggml_tensor* work;
    @cppsize(0) public ggml_tensor*[4096] nodes;
    @cppsize(0) public ggml_tensor*[4096] grads;
    @cppsize(0) public ggml_tensor*[4096] leafs;
    @cppsize(0) public int perf_runs;
    @cppsize(0) public int64_t perf_cycles;
    @cppsize(0) public int64_t perf_time_us;
}
extern(C)
@cppclasssize(24) align(8)
struct ggml_scratch
{


    @cppsize(0) public size_t offs;
    @cppsize(0) public size_t size;
    @cppsize(0) public void* data;
}
extern(C)
@cppclasssize(16) align(8)
struct ggml_init_params
{


    @cppsize(0) public size_t mem_size;
    @cppsize(0) public void* mem_buffer;
}
extern(C)
void ggml_time_init();

extern(C)
int64_t ggml_time_ms();

extern(C)
int64_t ggml_time_us();

extern(C)
int64_t ggml_cycles();

extern(C)
int64_t ggml_cycles_per_ms();

extern(C)
void ggml_print_object(const(ggml_object)* obj);

extern(C)
void ggml_print_objects(const(ggml_context)* ctx);

extern(C)
int ggml_nelements(const(ggml_tensor)* tensor);

extern(C)
size_t ggml_nbytes(const(ggml_tensor)* tensor);

extern(C)
int ggml_blck_size(ggml_type type);

extern(C)
size_t ggml_type_size(ggml_type type);

extern(C)
float ggml_type_sizef(ggml_type type);

extern(C)
size_t ggml_element_size(const(ggml_tensor)* tensor);

extern(C)
ggml_context* ggml_init(ggml_init_params params);

extern(C)
void ggml_free(ggml_context* ctx);

extern(C)
size_t ggml_used_mem(const(ggml_context)* ctx);

extern(C)
size_t ggml_set_scratch(ggml_context* ctx, ggml_scratch scratch);

extern(C)
ggml_tensor* ggml_new_tensor(ggml_context* ctx, ggml_type type, int n_dims, const(int)* ne);

extern(C)
ggml_tensor* ggml_new_tensor_1d(ggml_context* ctx, ggml_type type, int ne0);

extern(C)
ggml_tensor* ggml_new_tensor_2d(ggml_context* ctx, ggml_type type, int ne0, int ne1);

extern(C)
ggml_tensor* ggml_new_tensor_3d(ggml_context* ctx, ggml_type type, int ne0, int ne1, int ne2);

extern(C)
ggml_tensor* ggml_new_tensor_4d(ggml_context* ctx, ggml_type type, int ne0, int ne1, int ne2, int ne3);

extern(C)
ggml_tensor* ggml_new_i32(ggml_context* ctx, int32_t value);

extern(C)
ggml_tensor* ggml_new_f32(ggml_context* ctx, float value);

extern(C)
ggml_tensor* ggml_dup_tensor(ggml_context* ctx, const(ggml_tensor)* src);

extern(C)
ggml_tensor* ggml_view_tensor(ggml_context* ctx, const(ggml_tensor)* src);

extern(C)
ggml_tensor* ggml_set_zero(ggml_tensor* tensor);

extern(C)
ggml_tensor* ggml_set_i32(ggml_tensor* tensor, int32_t value);

extern(C)
ggml_tensor* ggml_set_f32(ggml_tensor* tensor, float value);

extern(C)
int32_t ggml_get_i32_1d(const(ggml_tensor)* tensor, int i);

extern(C)
void ggml_set_i32_1d(const(ggml_tensor)* tensor, int i, int32_t value);

extern(C)
float ggml_get_f32_1d(const(ggml_tensor)* tensor, int i);

extern(C)
void ggml_set_f32_1d(const(ggml_tensor)* tensor, int i, float value);

extern(C)
void* ggml_get_data(const(ggml_tensor)* tensor);

extern(C)
float* ggml_get_data_f32(const(ggml_tensor)* tensor);

extern(C)
ggml_tensor* ggml_dup(ggml_context* ctx, ggml_tensor* a);

extern(C)
ggml_tensor* ggml_add(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);

extern(C)
ggml_tensor* ggml_sub(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);

extern(C)
ggml_tensor* ggml_mul(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);

extern(C)
ggml_tensor* ggml_div(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);

extern(C)
ggml_tensor* ggml_sqr(ggml_context* ctx, ggml_tensor* a);

extern(C)
ggml_tensor* ggml_sqrt(ggml_context* ctx, ggml_tensor* a);

extern(C)
ggml_tensor* ggml_sum(ggml_context* ctx, ggml_tensor* a);

extern(C)
ggml_tensor* ggml_mean(ggml_context* ctx, ggml_tensor* a);

extern(C)
ggml_tensor* ggml_repeat(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);

extern(C)
ggml_tensor* ggml_abs(ggml_context* ctx, ggml_tensor* a);

extern(C)
ggml_tensor* ggml_sgn(ggml_context* ctx, ggml_tensor* a);

extern(C)
ggml_tensor* ggml_neg(ggml_context* ctx, ggml_tensor* a);

extern(C)
ggml_tensor* ggml_step(ggml_context* ctx, ggml_tensor* a);

extern(C)
ggml_tensor* ggml_relu(ggml_context* ctx, ggml_tensor* a);

extern(C)
ggml_tensor* ggml_gelu(ggml_context* ctx, ggml_tensor* a);

extern(C)
ggml_tensor* ggml_silu(ggml_context* ctx, ggml_tensor* a);

extern(C)
ggml_tensor* ggml_norm(ggml_context* ctx, ggml_tensor* a);

extern(C)
ggml_tensor* ggml_rms_norm(ggml_context* ctx, ggml_tensor* a);

extern(C)
ggml_tensor* ggml_mul_mat(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);

extern(C)
ggml_tensor* ggml_scale(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);

extern(C)
ggml_tensor* ggml_cpy(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);

extern(C)
ggml_tensor* ggml_reshape(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);

extern(C)
ggml_tensor* ggml_reshape_2d(ggml_context* ctx, ggml_tensor* a, int ne0, int ne1);

extern(C)
ggml_tensor* ggml_reshape_3d(ggml_context* ctx, ggml_tensor* a, int ne0, int ne1, int ne2);

extern(C)
ggml_tensor* ggml_view_1d(ggml_context* ctx, ggml_tensor* a, int ne0, size_t offset);

extern(C)
ggml_tensor* ggml_view_2d(ggml_context* ctx, ggml_tensor* a, int ne0, int ne1, size_t nb1, size_t offset);

extern(C)
ggml_tensor* ggml_permute(ggml_context* ctx, ggml_tensor* a, int axis0, int axis1, int axis2, int axis3);

extern(C)
ggml_tensor* ggml_transpose(ggml_context* ctx, ggml_tensor* a);

extern(C)
ggml_tensor* ggml_get_rows(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);

extern(C)
ggml_tensor* ggml_diag_mask_inf(ggml_context* ctx, ggml_tensor* a, int n_past);

extern(C)
ggml_tensor* ggml_soft_max(ggml_context* ctx, ggml_tensor* a);

extern(C)
ggml_tensor* ggml_rope(ggml_context* ctx, ggml_tensor* a, int n_past, int n_dims, int mode);

extern(C)
ggml_tensor* ggml_conv_1d_1s(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);

extern(C)
ggml_tensor* ggml_conv_1d_2s(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);

extern(C)
ggml_tensor* ggml_flash_attn(ggml_context* ctx, ggml_tensor* q, ggml_tensor* k, ggml_tensor* v, bool masked);

extern(C)
ggml_tensor* ggml_flash_ff(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b0, ggml_tensor* b1, ggml_tensor* c0, ggml_tensor* c1);

extern(C)
void ggml_set_param(ggml_context* ctx, ggml_tensor* tensor);

extern(C)
void ggml_build_forward_expand(ggml_cgraph* cgraph, ggml_tensor* tensor);

extern(C)
ggml_cgraph ggml_build_forward(ggml_tensor* tensor);

extern(C)
ggml_cgraph ggml_build_backward(ggml_context* ctx, ggml_cgraph* gf, bool keep);

extern(C)
void ggml_graph_compute(ggml_context* ctx, ggml_cgraph* cgraph);

extern(C)
void ggml_graph_reset(ggml_cgraph* cgraph);

extern(C)
void ggml_graph_print(const(ggml_cgraph)* cgraph);

extern(C)
void ggml_graph_dump_dot(const(ggml_cgraph)* gb, const(ggml_cgraph)* gf, const(char)* filename);

enum ggml_opt_type
{
    GGML_OPT_ADAM = 0, 
    GGML_OPT_LBFGS = 1, 
}

alias GGML_OPT_ADAM = ggml_opt_type.GGML_OPT_ADAM;
alias GGML_OPT_LBFGS = ggml_opt_type.GGML_OPT_LBFGS;

enum ggml_linesearch
{
    GGML_LINESEARCH_DEFAULT = 1, 
    GGML_LINESEARCH_BACKTRACKING_ARMIJO = 0, 
    GGML_LINESEARCH_BACKTRACKING_WOLFE = 1, 
    GGML_LINESEARCH_BACKTRACKING_STRONG_WOLFE = 2, 
}

alias GGML_LINESEARCH_DEFAULT = ggml_linesearch.GGML_LINESEARCH_DEFAULT;
alias GGML_LINESEARCH_BACKTRACKING_ARMIJO = ggml_linesearch.GGML_LINESEARCH_BACKTRACKING_ARMIJO;
alias GGML_LINESEARCH_BACKTRACKING_WOLFE = ggml_linesearch.GGML_LINESEARCH_BACKTRACKING_WOLFE;
alias GGML_LINESEARCH_BACKTRACKING_STRONG_WOLFE = ggml_linesearch.GGML_LINESEARCH_BACKTRACKING_STRONG_WOLFE;

enum ggml_opt_result
{
    GGML_OPT_OK = 0, 
    GGML_OPT_DID_NOT_CONVERGE = 1, 
    GGML_OPT_NO_CONTEXT = 2, 
    GGML_OPT_INVALID_WOLFE = 3, 
    GGML_OPT_FAIL = 4, 
    GGML_LINESEARCH_FAIL = -128, 
    GGML_LINESEARCH_MINIMUM_STEP = -127, 
    GGML_LINESEARCH_MAXIMUM_STEP = -126, 
    GGML_LINESEARCH_MAXIMUM_ITERATIONS = -125, 
    GGML_LINESEARCH_INVALID_PARAMETERS = -124, 
}

alias GGML_OPT_OK = ggml_opt_result.GGML_OPT_OK;
alias GGML_OPT_DID_NOT_CONVERGE = ggml_opt_result.GGML_OPT_DID_NOT_CONVERGE;
alias GGML_OPT_NO_CONTEXT = ggml_opt_result.GGML_OPT_NO_CONTEXT;
alias GGML_OPT_INVALID_WOLFE = ggml_opt_result.GGML_OPT_INVALID_WOLFE;
alias GGML_OPT_FAIL = ggml_opt_result.GGML_OPT_FAIL;
alias GGML_LINESEARCH_FAIL = ggml_opt_result.GGML_LINESEARCH_FAIL;
alias GGML_LINESEARCH_MINIMUM_STEP = ggml_opt_result.GGML_LINESEARCH_MINIMUM_STEP;
alias GGML_LINESEARCH_MAXIMUM_STEP = ggml_opt_result.GGML_LINESEARCH_MAXIMUM_STEP;
alias GGML_LINESEARCH_MAXIMUM_ITERATIONS = ggml_opt_result.GGML_LINESEARCH_MAXIMUM_ITERATIONS;
alias GGML_LINESEARCH_INVALID_PARAMETERS = ggml_opt_result.GGML_LINESEARCH_INVALID_PARAMETERS;

extern(C)
@cppclasssize(88) align(4)
struct ggml_opt_params
{


    extern(C)
    @cppclasssize(28) align(4)
    struct _anon1
    {
        @cppsize(0) public int n_iter;
        @cppsize(0) public float alpha;
        @cppsize(0) public float beta1;
        @cppsize(0) public float beta2;
        @cppsize(0) public float eps;
        @cppsize(0) public float eps_f;
        @cppsize(0) public float eps_g;
    }


    extern(C)
    @cppclasssize(36) align(4)
    struct _anon2
    {
        @cppsize(0) public int m;
        @cppsize(0) public int n_iter;
        @cppsize(0) public int max_linesearch;
        @cppsize(0) public float eps;
        @cppsize(0) public float ftol;
        @cppsize(0) public float wolfe;
        @cppsize(0) public float min_step;
        @cppsize(0) public float max_step;
        @cppsize(0) public ggml_linesearch linesearch;
    }


    @cppsize(0) public ggml_opt_type type;
    @cppsize(0) public int n_threads;
    @cppsize(0) public int past;
    @cppsize(0) public float delta;
    @cppsize(0) public int max_no_improvement;
    @cppsize(0) public bool print_forward_graph;
    @cppsize(0) public bool print_backward_graph;
    @cppsize(0) public _anon1 adam;
    @cppsize(0) public _anon2 lbfgs;
}
extern(C)
ggml_opt_params ggml_opt_default_params(ggml_opt_type type);

extern(C)
ggml_opt_result ggml_opt(ggml_context* ctx, ggml_opt_params params, ggml_tensor* f);

extern(C)
size_t ggml_quantize_q4_0(const(float)* src, void* dst, int n, int k, int qk, int64_t* hist);

extern(C)
size_t ggml_quantize_q4_1(const(float)* src, void* dst, int n, int k, int qk, int64_t* hist);

extern(C)
int ggml_cpu_has_avx();

extern(C)
int ggml_cpu_has_avx2();

extern(C)
int ggml_cpu_has_avx512();

extern(C)
int ggml_cpu_has_fma();

extern(C)
int ggml_cpu_has_neon();

extern(C)
int ggml_cpu_has_arm_fma();

extern(C)
int ggml_cpu_has_f16c();

extern(C)
int ggml_cpu_has_fp16_va();

extern(C)
int ggml_cpu_has_wasm_simd();

extern(C)
int ggml_cpu_has_blas();

extern(C)
int ggml_cpu_has_sse3();

extern(C)
int ggml_cpu_has_vsx();

enum LLAMA_FILE_VERSION = 1;
enum LLAMA_FILE_MAGIC = 0x67676d66;
enum LLAMA_FILE_MAGIC_UNVERSIONED = 0x67676d6c;
alias llama_token = int;

extern(C)
@cppclasssize(12) align(4)
struct llama_token_data
{


    @cppsize(0) public llama_token id;
    @cppsize(0) public float p;
    @cppsize(0) public float plog;
}
extern(C)
@cppclasssize(16) align(4)
struct llama_context_params
{


    @cppsize(0) public int n_ctx;
    @cppsize(0) public int n_parts;
    @cppsize(0) public int seed;
    @cppsize(0) public bool f16_kv;
    @cppsize(0) public bool logits_all;
    @cppsize(0) public bool vocab_only;
}
extern(C)
llama_context_params llama_context_default_params();

extern(C)
llama_context* llama_init_from_file(const(char)* path_model, llama_context_params params);

extern(C)
void llama_free(llama_context* ctx);

extern(C)
int llama_model_quantize(const(char)* fname_inp, const(char)* fname_out, int itype, int qk);

extern(C)
int llama_eval(llama_context* ctx, const(llama_token)* tokens, int n_tokens, int n_past, int n_threads);

extern(C)
int llama_tokenize(llama_context* ctx, const(char)* text, llama_token* tokens, int n_max_tokens, bool add_bos);

extern(C)
int llama_n_vocab(llama_context* ctx);

extern(C)
int llama_n_ctx(llama_context* ctx);

extern(C)
float* llama_get_logits(llama_context* ctx);

extern(C)
const(char)* llama_token_to_str(llama_context* ctx, llama_token token);

extern(C)
llama_token llama_token_bos();

extern(C)
llama_token llama_token_eos();

extern(C)
llama_token llama_sample_top_p_top_k(llama_context* ctx, const(llama_token)* last_n_tokens_data, int last_n_tokens_size, int top_k, double top_p, double temp, double repeat_penalty);

extern(C)
void llama_print_timings(llama_context* ctx);

extern(C)
void llama_reset_timings(llama_context* ctx);

extern(C)
const(char)* llama_print_system_info();

extern(C++)
@cppclasssize(144) align(8)
struct gpt_params
{


    @cppsize(0) public int32_t seed;
    @cppsize(0) public int32_t n_threads;
    @cppsize(0) public int32_t n_predict;
    @cppsize(0) public int32_t repeat_last_n;
    @cppsize(0) public int32_t n_parts;
    @cppsize(0) public int32_t n_ctx;
    @cppsize(0) public int32_t top_k;
    @cppsize(0) public float top_p;
    @cppsize(0) public float temp;
    @cppsize(0) public float repeat_penalty;
    @cppsize(0) public int32_t n_batch;
    @cppsize(0) public string model;
    @cppsize(0) public string prompt;
    @cppsize(0) public vector!(string) antiprompt;
    @cppsize(0) public bool memory_f16;
    @cppsize(0) public bool random_prompt;
    @cppsize(0) public bool use_color;
    @cppsize(0) public bool interactive;
    @cppsize(0) public bool interactive_start;
    @cppsize(0) public bool instruct;
    @cppsize(0) public bool ignore_eos;
    @cppsize(0) public bool perplexity;
}
extern(C++)
bool gpt_params_parse(int argc, char** argv, ref gpt_params params);

extern(C++)
void gpt_print_usage(int argc, char** argv, ref const(gpt_params) params);

extern(C++)
string gpt_random_prompt(ref mt19937 rng);

extern(C++)
vector!(llama_token) llama_tokenize(llama_context* ctx, ref const(string) text, bool add_bos);

struct llama_context;
struct ggml_context;
struct ggml_object;
