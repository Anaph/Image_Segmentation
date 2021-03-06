ý6
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8խ,
?
conv2d_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_16/kernel
}
$conv2d_16/kernel/Read/ReadVariableOpReadVariableOpconv2d_16/kernel*&
_output_shapes
: *
dtype0
t
conv2d_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_16/bias
m
"conv2d_16/bias/Read/ReadVariableOpReadVariableOpconv2d_16/bias*
_output_shapes
: *
dtype0
?
batch_normalization_15/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_15/gamma
?
0batch_normalization_15/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_15/gamma*
_output_shapes
: *
dtype0
?
batch_normalization_15/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_15/beta
?
/batch_normalization_15/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_15/beta*
_output_shapes
: *
dtype0
?
"batch_normalization_15/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_15/moving_mean
?
6batch_normalization_15/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_15/moving_mean*
_output_shapes
: *
dtype0
?
&batch_normalization_15/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_15/moving_variance
?
:batch_normalization_15/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_15/moving_variance*
_output_shapes
: *
dtype0
?
conv2d_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_18/kernel
}
$conv2d_18/kernel/Read/ReadVariableOpReadVariableOpconv2d_18/kernel*&
_output_shapes
: @*
dtype0
t
conv2d_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_18/bias
m
"conv2d_18/bias/Read/ReadVariableOpReadVariableOpconv2d_18/bias*
_output_shapes
:@*
dtype0
?
batch_normalization_17/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_17/gamma
?
0batch_normalization_17/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_17/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_17/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_17/beta
?
/batch_normalization_17/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_17/beta*
_output_shapes
:@*
dtype0
?
"batch_normalization_17/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_17/moving_mean
?
6batch_normalization_17/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_17/moving_mean*
_output_shapes
:@*
dtype0
?
&batch_normalization_17/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_17/moving_variance
?
:batch_normalization_17/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_17/moving_variance*
_output_shapes
:@*
dtype0
?
conv2d_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*!
shared_nameconv2d_20/kernel
~
$conv2d_20/kernel/Read/ReadVariableOpReadVariableOpconv2d_20/kernel*'
_output_shapes
:@?*
dtype0
u
conv2d_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_20/bias
n
"conv2d_20/bias/Read/ReadVariableOpReadVariableOpconv2d_20/bias*
_output_shapes	
:?*
dtype0
?
batch_normalization_19/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_19/gamma
?
0batch_normalization_19/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_19/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_19/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_19/beta
?
/batch_normalization_19/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_19/beta*
_output_shapes	
:?*
dtype0
?
"batch_normalization_19/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_19/moving_mean
?
6batch_normalization_19/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_19/moving_mean*
_output_shapes	
:?*
dtype0
?
&batch_normalization_19/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_19/moving_variance
?
:batch_normalization_19/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_19/moving_variance*
_output_shapes	
:?*
dtype0
?
conv2d_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*!
shared_nameconv2d_22/kernel

$conv2d_22/kernel/Read/ReadVariableOpReadVariableOpconv2d_22/kernel*(
_output_shapes
:??*
dtype0
u
conv2d_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_22/bias
n
"conv2d_22/bias/Read/ReadVariableOpReadVariableOpconv2d_22/bias*
_output_shapes	
:?*
dtype0
?
batch_normalization_21/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_21/gamma
?
0batch_normalization_21/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_21/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_21/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_21/beta
?
/batch_normalization_21/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_21/beta*
_output_shapes	
:?*
dtype0
?
"batch_normalization_21/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_21/moving_mean
?
6batch_normalization_21/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_21/moving_mean*
_output_shapes	
:?*
dtype0
?
&batch_normalization_21/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_21/moving_variance
?
:batch_normalization_21/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_21/moving_variance*
_output_shapes	
:?*
dtype0
?
conv2d_transpose_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??**
shared_nameconv2d_transpose_3/kernel
?
-conv2d_transpose_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_3/kernel*(
_output_shapes
:??*
dtype0
?
conv2d_transpose_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_nameconv2d_transpose_3/bias
?
+conv2d_transpose_3/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_3/bias*
_output_shapes	
:?*
dtype0
?
conv2d_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*!
shared_nameconv2d_24/kernel

$conv2d_24/kernel/Read/ReadVariableOpReadVariableOpconv2d_24/kernel*(
_output_shapes
:??*
dtype0
u
conv2d_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_24/bias
n
"conv2d_24/bias/Read/ReadVariableOpReadVariableOpconv2d_24/bias*
_output_shapes	
:?*
dtype0
?
batch_normalization_23/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_23/gamma
?
0batch_normalization_23/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_23/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_23/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_23/beta
?
/batch_normalization_23/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_23/beta*
_output_shapes	
:?*
dtype0
?
"batch_normalization_23/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_23/moving_mean
?
6batch_normalization_23/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_23/moving_mean*
_output_shapes	
:?*
dtype0
?
&batch_normalization_23/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_23/moving_variance
?
:batch_normalization_23/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_23/moving_variance*
_output_shapes	
:?*
dtype0
?
conv2d_transpose_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?**
shared_nameconv2d_transpose_4/kernel
?
-conv2d_transpose_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_4/kernel*'
_output_shapes
:@?*
dtype0
?
conv2d_transpose_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameconv2d_transpose_4/bias

+conv2d_transpose_4/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_4/bias*
_output_shapes
:@*
dtype0
?
conv2d_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?@*!
shared_nameconv2d_26/kernel
~
$conv2d_26/kernel/Read/ReadVariableOpReadVariableOpconv2d_26/kernel*'
_output_shapes
:?@*
dtype0
t
conv2d_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_26/bias
m
"conv2d_26/bias/Read/ReadVariableOpReadVariableOpconv2d_26/bias*
_output_shapes
:@*
dtype0
?
batch_normalization_25/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_25/gamma
?
0batch_normalization_25/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_25/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_25/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_25/beta
?
/batch_normalization_25/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_25/beta*
_output_shapes
:@*
dtype0
?
"batch_normalization_25/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_25/moving_mean
?
6batch_normalization_25/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_25/moving_mean*
_output_shapes
:@*
dtype0
?
&batch_normalization_25/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_25/moving_variance
?
:batch_normalization_25/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_25/moving_variance*
_output_shapes
:@*
dtype0
?
conv2d_transpose_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @**
shared_nameconv2d_transpose_5/kernel
?
-conv2d_transpose_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_5/kernel*&
_output_shapes
: @*
dtype0
?
conv2d_transpose_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameconv2d_transpose_5/bias

+conv2d_transpose_5/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_5/bias*
_output_shapes
: *
dtype0
?
conv2d_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *!
shared_nameconv2d_28/kernel
}
$conv2d_28/kernel/Read/ReadVariableOpReadVariableOpconv2d_28/kernel*&
_output_shapes
:@ *
dtype0
t
conv2d_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_28/bias
m
"conv2d_28/bias/Read/ReadVariableOpReadVariableOpconv2d_28/bias*
_output_shapes
: *
dtype0
?
batch_normalization_27/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_27/gamma
?
0batch_normalization_27/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_27/gamma*
_output_shapes
: *
dtype0
?
batch_normalization_27/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_27/beta
?
/batch_normalization_27/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_27/beta*
_output_shapes
: *
dtype0
?
"batch_normalization_27/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_27/moving_mean
?
6batch_normalization_27/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_27/moving_mean*
_output_shapes
: *
dtype0
?
&batch_normalization_27/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_27/moving_variance
?
:batch_normalization_27/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_27/moving_variance*
_output_shapes
: *
dtype0
?
conv2d_29/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_29/kernel
}
$conv2d_29/kernel/Read/ReadVariableOpReadVariableOpconv2d_29/kernel*&
_output_shapes
: *
dtype0
t
conv2d_29/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_29/bias
m
"conv2d_29/bias/Read/ReadVariableOpReadVariableOpconv2d_29/bias*
_output_shapes
:*
dtype0
t
cond_1/Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *!
shared_namecond_1/Adam/iter
m
$cond_1/Adam/iter/Read/ReadVariableOpReadVariableOpcond_1/Adam/iter*
_output_shapes
: *
dtype0	
x
cond_1/Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_namecond_1/Adam/beta_1
q
&cond_1/Adam/beta_1/Read/ReadVariableOpReadVariableOpcond_1/Adam/beta_1*
_output_shapes
: *
dtype0
x
cond_1/Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_namecond_1/Adam/beta_2
q
&cond_1/Adam/beta_2/Read/ReadVariableOpReadVariableOpcond_1/Adam/beta_2*
_output_shapes
: *
dtype0
v
cond_1/Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_namecond_1/Adam/decay
o
%cond_1/Adam/decay/Read/ReadVariableOpReadVariableOpcond_1/Adam/decay*
_output_shapes
: *
dtype0
?
cond_1/Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namecond_1/Adam/learning_rate

-cond_1/Adam/learning_rate/Read/ReadVariableOpReadVariableOpcond_1/Adam/learning_rate*
_output_shapes
: *
dtype0
x
current_loss_scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_namecurrent_loss_scale
q
&current_loss_scale/Read/ReadVariableOpReadVariableOpcurrent_loss_scale*
_output_shapes
: *
dtype0
h

good_stepsVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
good_steps
a
good_steps/Read/ReadVariableOpReadVariableOp
good_steps*
_output_shapes
: *
dtype0	
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
cond_1/Adam/conv2d_16/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name cond_1/Adam/conv2d_16/kernel/m
?
2cond_1/Adam/conv2d_16/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_16/kernel/m*&
_output_shapes
: *
dtype0
?
cond_1/Adam/conv2d_16/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namecond_1/Adam/conv2d_16/bias/m
?
0cond_1/Adam/conv2d_16/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_16/bias/m*
_output_shapes
: *
dtype0
?
*cond_1/Adam/batch_normalization_15/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*cond_1/Adam/batch_normalization_15/gamma/m
?
>cond_1/Adam/batch_normalization_15/gamma/m/Read/ReadVariableOpReadVariableOp*cond_1/Adam/batch_normalization_15/gamma/m*
_output_shapes
: *
dtype0
?
)cond_1/Adam/batch_normalization_15/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *:
shared_name+)cond_1/Adam/batch_normalization_15/beta/m
?
=cond_1/Adam/batch_normalization_15/beta/m/Read/ReadVariableOpReadVariableOp)cond_1/Adam/batch_normalization_15/beta/m*
_output_shapes
: *
dtype0
?
cond_1/Adam/conv2d_18/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*/
shared_name cond_1/Adam/conv2d_18/kernel/m
?
2cond_1/Adam/conv2d_18/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_18/kernel/m*&
_output_shapes
: @*
dtype0
?
cond_1/Adam/conv2d_18/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namecond_1/Adam/conv2d_18/bias/m
?
0cond_1/Adam/conv2d_18/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_18/bias/m*
_output_shapes
:@*
dtype0
?
*cond_1/Adam/batch_normalization_17/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*cond_1/Adam/batch_normalization_17/gamma/m
?
>cond_1/Adam/batch_normalization_17/gamma/m/Read/ReadVariableOpReadVariableOp*cond_1/Adam/batch_normalization_17/gamma/m*
_output_shapes
:@*
dtype0
?
)cond_1/Adam/batch_normalization_17/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)cond_1/Adam/batch_normalization_17/beta/m
?
=cond_1/Adam/batch_normalization_17/beta/m/Read/ReadVariableOpReadVariableOp)cond_1/Adam/batch_normalization_17/beta/m*
_output_shapes
:@*
dtype0
?
cond_1/Adam/conv2d_20/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*/
shared_name cond_1/Adam/conv2d_20/kernel/m
?
2cond_1/Adam/conv2d_20/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_20/kernel/m*'
_output_shapes
:@?*
dtype0
?
cond_1/Adam/conv2d_20/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namecond_1/Adam/conv2d_20/bias/m
?
0cond_1/Adam/conv2d_20/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_20/bias/m*
_output_shapes	
:?*
dtype0
?
*cond_1/Adam/batch_normalization_19/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*;
shared_name,*cond_1/Adam/batch_normalization_19/gamma/m
?
>cond_1/Adam/batch_normalization_19/gamma/m/Read/ReadVariableOpReadVariableOp*cond_1/Adam/batch_normalization_19/gamma/m*
_output_shapes	
:?*
dtype0
?
)cond_1/Adam/batch_normalization_19/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*:
shared_name+)cond_1/Adam/batch_normalization_19/beta/m
?
=cond_1/Adam/batch_normalization_19/beta/m/Read/ReadVariableOpReadVariableOp)cond_1/Adam/batch_normalization_19/beta/m*
_output_shapes	
:?*
dtype0
?
cond_1/Adam/conv2d_22/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*/
shared_name cond_1/Adam/conv2d_22/kernel/m
?
2cond_1/Adam/conv2d_22/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_22/kernel/m*(
_output_shapes
:??*
dtype0
?
cond_1/Adam/conv2d_22/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namecond_1/Adam/conv2d_22/bias/m
?
0cond_1/Adam/conv2d_22/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_22/bias/m*
_output_shapes	
:?*
dtype0
?
*cond_1/Adam/batch_normalization_21/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*;
shared_name,*cond_1/Adam/batch_normalization_21/gamma/m
?
>cond_1/Adam/batch_normalization_21/gamma/m/Read/ReadVariableOpReadVariableOp*cond_1/Adam/batch_normalization_21/gamma/m*
_output_shapes	
:?*
dtype0
?
)cond_1/Adam/batch_normalization_21/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*:
shared_name+)cond_1/Adam/batch_normalization_21/beta/m
?
=cond_1/Adam/batch_normalization_21/beta/m/Read/ReadVariableOpReadVariableOp)cond_1/Adam/batch_normalization_21/beta/m*
_output_shapes	
:?*
dtype0
?
'cond_1/Adam/conv2d_transpose_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*8
shared_name)'cond_1/Adam/conv2d_transpose_3/kernel/m
?
;cond_1/Adam/conv2d_transpose_3/kernel/m/Read/ReadVariableOpReadVariableOp'cond_1/Adam/conv2d_transpose_3/kernel/m*(
_output_shapes
:??*
dtype0
?
%cond_1/Adam/conv2d_transpose_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*6
shared_name'%cond_1/Adam/conv2d_transpose_3/bias/m
?
9cond_1/Adam/conv2d_transpose_3/bias/m/Read/ReadVariableOpReadVariableOp%cond_1/Adam/conv2d_transpose_3/bias/m*
_output_shapes	
:?*
dtype0
?
cond_1/Adam/conv2d_24/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*/
shared_name cond_1/Adam/conv2d_24/kernel/m
?
2cond_1/Adam/conv2d_24/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_24/kernel/m*(
_output_shapes
:??*
dtype0
?
cond_1/Adam/conv2d_24/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namecond_1/Adam/conv2d_24/bias/m
?
0cond_1/Adam/conv2d_24/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_24/bias/m*
_output_shapes	
:?*
dtype0
?
*cond_1/Adam/batch_normalization_23/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*;
shared_name,*cond_1/Adam/batch_normalization_23/gamma/m
?
>cond_1/Adam/batch_normalization_23/gamma/m/Read/ReadVariableOpReadVariableOp*cond_1/Adam/batch_normalization_23/gamma/m*
_output_shapes	
:?*
dtype0
?
)cond_1/Adam/batch_normalization_23/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*:
shared_name+)cond_1/Adam/batch_normalization_23/beta/m
?
=cond_1/Adam/batch_normalization_23/beta/m/Read/ReadVariableOpReadVariableOp)cond_1/Adam/batch_normalization_23/beta/m*
_output_shapes	
:?*
dtype0
?
'cond_1/Adam/conv2d_transpose_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*8
shared_name)'cond_1/Adam/conv2d_transpose_4/kernel/m
?
;cond_1/Adam/conv2d_transpose_4/kernel/m/Read/ReadVariableOpReadVariableOp'cond_1/Adam/conv2d_transpose_4/kernel/m*'
_output_shapes
:@?*
dtype0
?
%cond_1/Adam/conv2d_transpose_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%cond_1/Adam/conv2d_transpose_4/bias/m
?
9cond_1/Adam/conv2d_transpose_4/bias/m/Read/ReadVariableOpReadVariableOp%cond_1/Adam/conv2d_transpose_4/bias/m*
_output_shapes
:@*
dtype0
?
cond_1/Adam/conv2d_26/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?@*/
shared_name cond_1/Adam/conv2d_26/kernel/m
?
2cond_1/Adam/conv2d_26/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_26/kernel/m*'
_output_shapes
:?@*
dtype0
?
cond_1/Adam/conv2d_26/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namecond_1/Adam/conv2d_26/bias/m
?
0cond_1/Adam/conv2d_26/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_26/bias/m*
_output_shapes
:@*
dtype0
?
*cond_1/Adam/batch_normalization_25/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*cond_1/Adam/batch_normalization_25/gamma/m
?
>cond_1/Adam/batch_normalization_25/gamma/m/Read/ReadVariableOpReadVariableOp*cond_1/Adam/batch_normalization_25/gamma/m*
_output_shapes
:@*
dtype0
?
)cond_1/Adam/batch_normalization_25/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)cond_1/Adam/batch_normalization_25/beta/m
?
=cond_1/Adam/batch_normalization_25/beta/m/Read/ReadVariableOpReadVariableOp)cond_1/Adam/batch_normalization_25/beta/m*
_output_shapes
:@*
dtype0
?
'cond_1/Adam/conv2d_transpose_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*8
shared_name)'cond_1/Adam/conv2d_transpose_5/kernel/m
?
;cond_1/Adam/conv2d_transpose_5/kernel/m/Read/ReadVariableOpReadVariableOp'cond_1/Adam/conv2d_transpose_5/kernel/m*&
_output_shapes
: @*
dtype0
?
%cond_1/Adam/conv2d_transpose_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%cond_1/Adam/conv2d_transpose_5/bias/m
?
9cond_1/Adam/conv2d_transpose_5/bias/m/Read/ReadVariableOpReadVariableOp%cond_1/Adam/conv2d_transpose_5/bias/m*
_output_shapes
: *
dtype0
?
cond_1/Adam/conv2d_28/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ */
shared_name cond_1/Adam/conv2d_28/kernel/m
?
2cond_1/Adam/conv2d_28/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_28/kernel/m*&
_output_shapes
:@ *
dtype0
?
cond_1/Adam/conv2d_28/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namecond_1/Adam/conv2d_28/bias/m
?
0cond_1/Adam/conv2d_28/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_28/bias/m*
_output_shapes
: *
dtype0
?
*cond_1/Adam/batch_normalization_27/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*cond_1/Adam/batch_normalization_27/gamma/m
?
>cond_1/Adam/batch_normalization_27/gamma/m/Read/ReadVariableOpReadVariableOp*cond_1/Adam/batch_normalization_27/gamma/m*
_output_shapes
: *
dtype0
?
)cond_1/Adam/batch_normalization_27/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *:
shared_name+)cond_1/Adam/batch_normalization_27/beta/m
?
=cond_1/Adam/batch_normalization_27/beta/m/Read/ReadVariableOpReadVariableOp)cond_1/Adam/batch_normalization_27/beta/m*
_output_shapes
: *
dtype0
?
cond_1/Adam/conv2d_29/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name cond_1/Adam/conv2d_29/kernel/m
?
2cond_1/Adam/conv2d_29/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_29/kernel/m*&
_output_shapes
: *
dtype0
?
cond_1/Adam/conv2d_29/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namecond_1/Adam/conv2d_29/bias/m
?
0cond_1/Adam/conv2d_29/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_29/bias/m*
_output_shapes
:*
dtype0
?
cond_1/Adam/conv2d_16/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name cond_1/Adam/conv2d_16/kernel/v
?
2cond_1/Adam/conv2d_16/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_16/kernel/v*&
_output_shapes
: *
dtype0
?
cond_1/Adam/conv2d_16/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namecond_1/Adam/conv2d_16/bias/v
?
0cond_1/Adam/conv2d_16/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_16/bias/v*
_output_shapes
: *
dtype0
?
*cond_1/Adam/batch_normalization_15/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*cond_1/Adam/batch_normalization_15/gamma/v
?
>cond_1/Adam/batch_normalization_15/gamma/v/Read/ReadVariableOpReadVariableOp*cond_1/Adam/batch_normalization_15/gamma/v*
_output_shapes
: *
dtype0
?
)cond_1/Adam/batch_normalization_15/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *:
shared_name+)cond_1/Adam/batch_normalization_15/beta/v
?
=cond_1/Adam/batch_normalization_15/beta/v/Read/ReadVariableOpReadVariableOp)cond_1/Adam/batch_normalization_15/beta/v*
_output_shapes
: *
dtype0
?
cond_1/Adam/conv2d_18/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*/
shared_name cond_1/Adam/conv2d_18/kernel/v
?
2cond_1/Adam/conv2d_18/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_18/kernel/v*&
_output_shapes
: @*
dtype0
?
cond_1/Adam/conv2d_18/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namecond_1/Adam/conv2d_18/bias/v
?
0cond_1/Adam/conv2d_18/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_18/bias/v*
_output_shapes
:@*
dtype0
?
*cond_1/Adam/batch_normalization_17/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*cond_1/Adam/batch_normalization_17/gamma/v
?
>cond_1/Adam/batch_normalization_17/gamma/v/Read/ReadVariableOpReadVariableOp*cond_1/Adam/batch_normalization_17/gamma/v*
_output_shapes
:@*
dtype0
?
)cond_1/Adam/batch_normalization_17/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)cond_1/Adam/batch_normalization_17/beta/v
?
=cond_1/Adam/batch_normalization_17/beta/v/Read/ReadVariableOpReadVariableOp)cond_1/Adam/batch_normalization_17/beta/v*
_output_shapes
:@*
dtype0
?
cond_1/Adam/conv2d_20/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*/
shared_name cond_1/Adam/conv2d_20/kernel/v
?
2cond_1/Adam/conv2d_20/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_20/kernel/v*'
_output_shapes
:@?*
dtype0
?
cond_1/Adam/conv2d_20/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namecond_1/Adam/conv2d_20/bias/v
?
0cond_1/Adam/conv2d_20/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_20/bias/v*
_output_shapes	
:?*
dtype0
?
*cond_1/Adam/batch_normalization_19/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*;
shared_name,*cond_1/Adam/batch_normalization_19/gamma/v
?
>cond_1/Adam/batch_normalization_19/gamma/v/Read/ReadVariableOpReadVariableOp*cond_1/Adam/batch_normalization_19/gamma/v*
_output_shapes	
:?*
dtype0
?
)cond_1/Adam/batch_normalization_19/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*:
shared_name+)cond_1/Adam/batch_normalization_19/beta/v
?
=cond_1/Adam/batch_normalization_19/beta/v/Read/ReadVariableOpReadVariableOp)cond_1/Adam/batch_normalization_19/beta/v*
_output_shapes	
:?*
dtype0
?
cond_1/Adam/conv2d_22/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*/
shared_name cond_1/Adam/conv2d_22/kernel/v
?
2cond_1/Adam/conv2d_22/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_22/kernel/v*(
_output_shapes
:??*
dtype0
?
cond_1/Adam/conv2d_22/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namecond_1/Adam/conv2d_22/bias/v
?
0cond_1/Adam/conv2d_22/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_22/bias/v*
_output_shapes	
:?*
dtype0
?
*cond_1/Adam/batch_normalization_21/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*;
shared_name,*cond_1/Adam/batch_normalization_21/gamma/v
?
>cond_1/Adam/batch_normalization_21/gamma/v/Read/ReadVariableOpReadVariableOp*cond_1/Adam/batch_normalization_21/gamma/v*
_output_shapes	
:?*
dtype0
?
)cond_1/Adam/batch_normalization_21/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*:
shared_name+)cond_1/Adam/batch_normalization_21/beta/v
?
=cond_1/Adam/batch_normalization_21/beta/v/Read/ReadVariableOpReadVariableOp)cond_1/Adam/batch_normalization_21/beta/v*
_output_shapes	
:?*
dtype0
?
'cond_1/Adam/conv2d_transpose_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*8
shared_name)'cond_1/Adam/conv2d_transpose_3/kernel/v
?
;cond_1/Adam/conv2d_transpose_3/kernel/v/Read/ReadVariableOpReadVariableOp'cond_1/Adam/conv2d_transpose_3/kernel/v*(
_output_shapes
:??*
dtype0
?
%cond_1/Adam/conv2d_transpose_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*6
shared_name'%cond_1/Adam/conv2d_transpose_3/bias/v
?
9cond_1/Adam/conv2d_transpose_3/bias/v/Read/ReadVariableOpReadVariableOp%cond_1/Adam/conv2d_transpose_3/bias/v*
_output_shapes	
:?*
dtype0
?
cond_1/Adam/conv2d_24/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*/
shared_name cond_1/Adam/conv2d_24/kernel/v
?
2cond_1/Adam/conv2d_24/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_24/kernel/v*(
_output_shapes
:??*
dtype0
?
cond_1/Adam/conv2d_24/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namecond_1/Adam/conv2d_24/bias/v
?
0cond_1/Adam/conv2d_24/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_24/bias/v*
_output_shapes	
:?*
dtype0
?
*cond_1/Adam/batch_normalization_23/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*;
shared_name,*cond_1/Adam/batch_normalization_23/gamma/v
?
>cond_1/Adam/batch_normalization_23/gamma/v/Read/ReadVariableOpReadVariableOp*cond_1/Adam/batch_normalization_23/gamma/v*
_output_shapes	
:?*
dtype0
?
)cond_1/Adam/batch_normalization_23/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*:
shared_name+)cond_1/Adam/batch_normalization_23/beta/v
?
=cond_1/Adam/batch_normalization_23/beta/v/Read/ReadVariableOpReadVariableOp)cond_1/Adam/batch_normalization_23/beta/v*
_output_shapes	
:?*
dtype0
?
'cond_1/Adam/conv2d_transpose_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*8
shared_name)'cond_1/Adam/conv2d_transpose_4/kernel/v
?
;cond_1/Adam/conv2d_transpose_4/kernel/v/Read/ReadVariableOpReadVariableOp'cond_1/Adam/conv2d_transpose_4/kernel/v*'
_output_shapes
:@?*
dtype0
?
%cond_1/Adam/conv2d_transpose_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%cond_1/Adam/conv2d_transpose_4/bias/v
?
9cond_1/Adam/conv2d_transpose_4/bias/v/Read/ReadVariableOpReadVariableOp%cond_1/Adam/conv2d_transpose_4/bias/v*
_output_shapes
:@*
dtype0
?
cond_1/Adam/conv2d_26/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?@*/
shared_name cond_1/Adam/conv2d_26/kernel/v
?
2cond_1/Adam/conv2d_26/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_26/kernel/v*'
_output_shapes
:?@*
dtype0
?
cond_1/Adam/conv2d_26/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namecond_1/Adam/conv2d_26/bias/v
?
0cond_1/Adam/conv2d_26/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_26/bias/v*
_output_shapes
:@*
dtype0
?
*cond_1/Adam/batch_normalization_25/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*cond_1/Adam/batch_normalization_25/gamma/v
?
>cond_1/Adam/batch_normalization_25/gamma/v/Read/ReadVariableOpReadVariableOp*cond_1/Adam/batch_normalization_25/gamma/v*
_output_shapes
:@*
dtype0
?
)cond_1/Adam/batch_normalization_25/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)cond_1/Adam/batch_normalization_25/beta/v
?
=cond_1/Adam/batch_normalization_25/beta/v/Read/ReadVariableOpReadVariableOp)cond_1/Adam/batch_normalization_25/beta/v*
_output_shapes
:@*
dtype0
?
'cond_1/Adam/conv2d_transpose_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*8
shared_name)'cond_1/Adam/conv2d_transpose_5/kernel/v
?
;cond_1/Adam/conv2d_transpose_5/kernel/v/Read/ReadVariableOpReadVariableOp'cond_1/Adam/conv2d_transpose_5/kernel/v*&
_output_shapes
: @*
dtype0
?
%cond_1/Adam/conv2d_transpose_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%cond_1/Adam/conv2d_transpose_5/bias/v
?
9cond_1/Adam/conv2d_transpose_5/bias/v/Read/ReadVariableOpReadVariableOp%cond_1/Adam/conv2d_transpose_5/bias/v*
_output_shapes
: *
dtype0
?
cond_1/Adam/conv2d_28/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ */
shared_name cond_1/Adam/conv2d_28/kernel/v
?
2cond_1/Adam/conv2d_28/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_28/kernel/v*&
_output_shapes
:@ *
dtype0
?
cond_1/Adam/conv2d_28/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namecond_1/Adam/conv2d_28/bias/v
?
0cond_1/Adam/conv2d_28/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_28/bias/v*
_output_shapes
: *
dtype0
?
*cond_1/Adam/batch_normalization_27/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*cond_1/Adam/batch_normalization_27/gamma/v
?
>cond_1/Adam/batch_normalization_27/gamma/v/Read/ReadVariableOpReadVariableOp*cond_1/Adam/batch_normalization_27/gamma/v*
_output_shapes
: *
dtype0
?
)cond_1/Adam/batch_normalization_27/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *:
shared_name+)cond_1/Adam/batch_normalization_27/beta/v
?
=cond_1/Adam/batch_normalization_27/beta/v/Read/ReadVariableOpReadVariableOp)cond_1/Adam/batch_normalization_27/beta/v*
_output_shapes
: *
dtype0
?
cond_1/Adam/conv2d_29/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name cond_1/Adam/conv2d_29/kernel/v
?
2cond_1/Adam/conv2d_29/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_29/kernel/v*&
_output_shapes
: *
dtype0
?
cond_1/Adam/conv2d_29/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namecond_1/Adam/conv2d_29/bias/v
?
0cond_1/Adam/conv2d_29/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_29/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer-8

layer-9
layer-10
layer_with_weights-4
layer-11
layer_with_weights-5
layer-12
layer-13
layer-14
layer-15
layer_with_weights-6
layer-16
layer_with_weights-7
layer-17
layer-18
layer_with_weights-8
layer-19
layer-20
layer-21
layer_with_weights-9
layer-22
layer_with_weights-10
layer-23
layer-24
layer_with_weights-11
layer-25
layer-26
layer-27
layer_with_weights-12
layer-28
layer_with_weights-13
layer-29
layer-30
 layer_with_weights-14
 layer-31
!layer-32
"layer-33
#layer_with_weights-15
#layer-34
$layer_with_weights-16
$layer-35
%layer-36
&layer_with_weights-17
&layer-37
'	optimizer
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,
signatures
 
h

-kernel
.bias
/regularization_losses
0	variables
1trainable_variables
2	keras_api
?
3axis
	4gamma
5beta
6moving_mean
7moving_variance
8regularization_losses
9	variables
:trainable_variables
;	keras_api
R
<regularization_losses
=	variables
>trainable_variables
?	keras_api
R
@regularization_losses
A	variables
Btrainable_variables
C	keras_api
R
Dregularization_losses
E	variables
Ftrainable_variables
G	keras_api
h

Hkernel
Ibias
Jregularization_losses
K	variables
Ltrainable_variables
M	keras_api
?
Naxis
	Ogamma
Pbeta
Qmoving_mean
Rmoving_variance
Sregularization_losses
T	variables
Utrainable_variables
V	keras_api
R
Wregularization_losses
X	variables
Ytrainable_variables
Z	keras_api
R
[regularization_losses
\	variables
]trainable_variables
^	keras_api
R
_regularization_losses
`	variables
atrainable_variables
b	keras_api
h

ckernel
dbias
eregularization_losses
f	variables
gtrainable_variables
h	keras_api
?
iaxis
	jgamma
kbeta
lmoving_mean
mmoving_variance
nregularization_losses
o	variables
ptrainable_variables
q	keras_api
R
rregularization_losses
s	variables
ttrainable_variables
u	keras_api
R
vregularization_losses
w	variables
xtrainable_variables
y	keras_api
R
zregularization_losses
{	variables
|trainable_variables
}	keras_api
l

~kernel
bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?	variables
?trainable_variables
?	keras_api
V
?regularization_losses
?	variables
?trainable_variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
V
?regularization_losses
?	variables
?trainable_variables
?	keras_api
V
?regularization_losses
?	variables
?trainable_variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?	variables
?trainable_variables
?	keras_api
V
?regularization_losses
?	variables
?trainable_variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
V
?regularization_losses
?	variables
?trainable_variables
?	keras_api
V
?regularization_losses
?	variables
?trainable_variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?	variables
?trainable_variables
?	keras_api
V
?regularization_losses
?	variables
?trainable_variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
V
?regularization_losses
?	variables
?trainable_variables
?	keras_api
V
?regularization_losses
?	variables
?trainable_variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?	variables
?trainable_variables
?	keras_api
V
?regularization_losses
?	variables
?trainable_variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?
?
loss_scale
?base_optimizer
	?iter
?beta_1
?beta_2

?decay
?learning_rate-m?.m?4m?5m?Hm?Im?Om?Pm?cm?dm?jm?km?~m?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?-v?.v?4v?5v?Hv?Iv?Ov?Pv?cv?dv?jv?kv?~v?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?
?
-0
.1
42
53
64
75
H6
I7
O8
P9
Q10
R11
c12
d13
j14
k15
l16
m17
~18
19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45
?46
?47
?48
?49
?
-0
.1
42
53
H4
I5
O6
P7
c8
d9
j10
k11
~12
13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
 
?
?layers
(	variables
)trainable_variables
*regularization_losses
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
?metrics
 
\Z
VARIABLE_VALUEconv2d_16/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_16/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

-0
.1

-0
.1
?
/regularization_losses
0	variables
1trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_15/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_15/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_15/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_15/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

40
51
62
73

40
51
?
8regularization_losses
9	variables
:trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
 
 
 
?
<regularization_losses
=	variables
>trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
 
 
 
?
@regularization_losses
A	variables
Btrainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
 
 
 
?
Dregularization_losses
E	variables
Ftrainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
\Z
VARIABLE_VALUEconv2d_18/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_18/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

H0
I1

H0
I1
?
Jregularization_losses
K	variables
Ltrainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_17/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_17/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_17/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_17/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

O0
P1
Q2
R3

O0
P1
?
Sregularization_losses
T	variables
Utrainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
 
 
 
?
Wregularization_losses
X	variables
Ytrainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
 
 
 
?
[regularization_losses
\	variables
]trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
 
 
 
?
_regularization_losses
`	variables
atrainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
\Z
VARIABLE_VALUEconv2d_20/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_20/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

c0
d1

c0
d1
?
eregularization_losses
f	variables
gtrainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_19/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_19/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_19/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_19/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

j0
k1
l2
m3

j0
k1
?
nregularization_losses
o	variables
ptrainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
 
 
 
?
rregularization_losses
s	variables
ttrainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
 
 
 
?
vregularization_losses
w	variables
xtrainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
 
 
 
?
zregularization_losses
{	variables
|trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
\Z
VARIABLE_VALUEconv2d_22/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_22/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

~0
1

~0
1
?
?regularization_losses
?	variables
?trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_21/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_21/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_21/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_21/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
 
?0
?1
?2
?3

?0
?1
?
?regularization_losses
?	variables
?trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
 
 
 
?
?regularization_losses
?	variables
?trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
ec
VARIABLE_VALUEconv2d_transpose_3/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEconv2d_transpose_3/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?regularization_losses
?	variables
?trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
 
 
 
?
?regularization_losses
?	variables
?trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
 
 
 
?
?regularization_losses
?	variables
?trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
\Z
VARIABLE_VALUEconv2d_24/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_24/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?regularization_losses
?	variables
?trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
 
hf
VARIABLE_VALUEbatch_normalization_23/gamma6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_23/beta5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_23/moving_mean<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_23/moving_variance@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
 
?0
?1
?2
?3

?0
?1
?
?regularization_losses
?	variables
?trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
 
 
 
?
?regularization_losses
?	variables
?trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
fd
VARIABLE_VALUEconv2d_transpose_4/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEconv2d_transpose_4/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?regularization_losses
?	variables
?trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
 
 
 
?
?regularization_losses
?	variables
?trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
 
 
 
?
?regularization_losses
?	variables
?trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
][
VARIABLE_VALUEconv2d_26/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_26/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?regularization_losses
?	variables
?trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
 
hf
VARIABLE_VALUEbatch_normalization_25/gamma6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_25/beta5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_25/moving_mean<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_25/moving_variance@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
 
?0
?1
?2
?3

?0
?1
?
?regularization_losses
?	variables
?trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
 
 
 
?
?regularization_losses
?	variables
?trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
fd
VARIABLE_VALUEconv2d_transpose_5/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEconv2d_transpose_5/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?regularization_losses
?	variables
?trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
 
 
 
?
?regularization_losses
?	variables
?trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
 
 
 
?
?regularization_losses
?	variables
?trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
][
VARIABLE_VALUEconv2d_28/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_28/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?regularization_losses
?	variables
?trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
 
hf
VARIABLE_VALUEbatch_normalization_27/gamma6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_27/beta5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_27/moving_mean<layer_with_weights-16/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_27/moving_variance@layer_with_weights-16/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
 
?0
?1
?2
?3

?0
?1
?
?regularization_losses
?	variables
?trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
 
 
 
?
?regularization_losses
?	variables
?trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
][
VARIABLE_VALUEconv2d_29/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_29/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?regularization_losses
?	variables
?trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
 
 
OM
VARIABLE_VALUEcond_1/Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEcond_1/Adam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEcond_1/Adam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcond_1/Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEcond_1/Adam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
 
 
n
60
71
Q2
R3
l4
m5
?6
?7
?8
?9
?10
?11
?12
?13

?0
?1
 
 
 
 
 
 
 
 
 

60
71
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

Q0
R1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

l0
m1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
 
 
jh
VARIABLE_VALUEcurrent_loss_scaleBoptimizer/loss_scale/current_loss_scale/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUE
good_steps:optimizer/loss_scale/good_steps/.ATTRIBUTES/VARIABLE_VALUE
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
??
VARIABLE_VALUEcond_1/Adam/conv2d_16/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_16/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*cond_1/Adam/batch_normalization_15/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)cond_1/Adam/batch_normalization_15/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_18/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_18/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*cond_1/Adam/batch_normalization_17/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)cond_1/Adam/batch_normalization_17/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_20/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_20/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*cond_1/Adam/batch_normalization_19/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)cond_1/Adam/batch_normalization_19/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_22/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_22/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*cond_1/Adam/batch_normalization_21/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)cond_1/Adam/batch_normalization_21/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE'cond_1/Adam/conv2d_transpose_3/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE%cond_1/Adam/conv2d_transpose_3/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_24/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_24/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*cond_1/Adam/batch_normalization_23/gamma/mRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)cond_1/Adam/batch_normalization_23/beta/mQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE'cond_1/Adam/conv2d_transpose_4/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE%cond_1/Adam/conv2d_transpose_4/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_26/kernel/mSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_26/bias/mQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*cond_1/Adam/batch_normalization_25/gamma/mRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)cond_1/Adam/batch_normalization_25/beta/mQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE'cond_1/Adam/conv2d_transpose_5/kernel/mSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE%cond_1/Adam/conv2d_transpose_5/bias/mQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_28/kernel/mSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_28/bias/mQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*cond_1/Adam/batch_normalization_27/gamma/mRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)cond_1/Adam/batch_normalization_27/beta/mQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_29/kernel/mSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_29/bias/mQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_16/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_16/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*cond_1/Adam/batch_normalization_15/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)cond_1/Adam/batch_normalization_15/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_18/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_18/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*cond_1/Adam/batch_normalization_17/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)cond_1/Adam/batch_normalization_17/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_20/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_20/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*cond_1/Adam/batch_normalization_19/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)cond_1/Adam/batch_normalization_19/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_22/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_22/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*cond_1/Adam/batch_normalization_21/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)cond_1/Adam/batch_normalization_21/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE'cond_1/Adam/conv2d_transpose_3/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE%cond_1/Adam/conv2d_transpose_3/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_24/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_24/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*cond_1/Adam/batch_normalization_23/gamma/vRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)cond_1/Adam/batch_normalization_23/beta/vQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE'cond_1/Adam/conv2d_transpose_4/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE%cond_1/Adam/conv2d_transpose_4/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_26/kernel/vSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_26/bias/vQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*cond_1/Adam/batch_normalization_25/gamma/vRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)cond_1/Adam/batch_normalization_25/beta/vQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE'cond_1/Adam/conv2d_transpose_5/kernel/vSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE%cond_1/Adam/conv2d_transpose_5/bias/vQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_28/kernel/vSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_28/bias/vQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*cond_1/Adam/batch_normalization_27/gamma/vRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)cond_1/Adam/batch_normalization_27/beta/vQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_29/kernel/vSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_29/bias/vQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_2Placeholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2conv2d_16/kernelconv2d_16/biasbatch_normalization_15/gammabatch_normalization_15/beta"batch_normalization_15/moving_mean&batch_normalization_15/moving_varianceconv2d_18/kernelconv2d_18/biasbatch_normalization_17/gammabatch_normalization_17/beta"batch_normalization_17/moving_mean&batch_normalization_17/moving_varianceconv2d_20/kernelconv2d_20/biasbatch_normalization_19/gammabatch_normalization_19/beta"batch_normalization_19/moving_mean&batch_normalization_19/moving_varianceconv2d_22/kernelconv2d_22/biasbatch_normalization_21/gammabatch_normalization_21/beta"batch_normalization_21/moving_mean&batch_normalization_21/moving_varianceconv2d_transpose_3/kernelconv2d_transpose_3/biasconv2d_24/kernelconv2d_24/biasbatch_normalization_23/gammabatch_normalization_23/beta"batch_normalization_23/moving_mean&batch_normalization_23/moving_varianceconv2d_transpose_4/kernelconv2d_transpose_4/biasconv2d_26/kernelconv2d_26/biasbatch_normalization_25/gammabatch_normalization_25/beta"batch_normalization_25/moving_mean&batch_normalization_25/moving_varianceconv2d_transpose_5/kernelconv2d_transpose_5/biasconv2d_28/kernelconv2d_28/biasbatch_normalization_27/gammabatch_normalization_27/beta"batch_normalization_27/moving_mean&batch_normalization_27/moving_varianceconv2d_29/kernelconv2d_29/bias*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./012*8
config_proto(&

CPU

GPU2*0J

  ?E8? *,
f'R%
#__inference_signature_wrapper_40263
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?9
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_16/kernel/Read/ReadVariableOp"conv2d_16/bias/Read/ReadVariableOp0batch_normalization_15/gamma/Read/ReadVariableOp/batch_normalization_15/beta/Read/ReadVariableOp6batch_normalization_15/moving_mean/Read/ReadVariableOp:batch_normalization_15/moving_variance/Read/ReadVariableOp$conv2d_18/kernel/Read/ReadVariableOp"conv2d_18/bias/Read/ReadVariableOp0batch_normalization_17/gamma/Read/ReadVariableOp/batch_normalization_17/beta/Read/ReadVariableOp6batch_normalization_17/moving_mean/Read/ReadVariableOp:batch_normalization_17/moving_variance/Read/ReadVariableOp$conv2d_20/kernel/Read/ReadVariableOp"conv2d_20/bias/Read/ReadVariableOp0batch_normalization_19/gamma/Read/ReadVariableOp/batch_normalization_19/beta/Read/ReadVariableOp6batch_normalization_19/moving_mean/Read/ReadVariableOp:batch_normalization_19/moving_variance/Read/ReadVariableOp$conv2d_22/kernel/Read/ReadVariableOp"conv2d_22/bias/Read/ReadVariableOp0batch_normalization_21/gamma/Read/ReadVariableOp/batch_normalization_21/beta/Read/ReadVariableOp6batch_normalization_21/moving_mean/Read/ReadVariableOp:batch_normalization_21/moving_variance/Read/ReadVariableOp-conv2d_transpose_3/kernel/Read/ReadVariableOp+conv2d_transpose_3/bias/Read/ReadVariableOp$conv2d_24/kernel/Read/ReadVariableOp"conv2d_24/bias/Read/ReadVariableOp0batch_normalization_23/gamma/Read/ReadVariableOp/batch_normalization_23/beta/Read/ReadVariableOp6batch_normalization_23/moving_mean/Read/ReadVariableOp:batch_normalization_23/moving_variance/Read/ReadVariableOp-conv2d_transpose_4/kernel/Read/ReadVariableOp+conv2d_transpose_4/bias/Read/ReadVariableOp$conv2d_26/kernel/Read/ReadVariableOp"conv2d_26/bias/Read/ReadVariableOp0batch_normalization_25/gamma/Read/ReadVariableOp/batch_normalization_25/beta/Read/ReadVariableOp6batch_normalization_25/moving_mean/Read/ReadVariableOp:batch_normalization_25/moving_variance/Read/ReadVariableOp-conv2d_transpose_5/kernel/Read/ReadVariableOp+conv2d_transpose_5/bias/Read/ReadVariableOp$conv2d_28/kernel/Read/ReadVariableOp"conv2d_28/bias/Read/ReadVariableOp0batch_normalization_27/gamma/Read/ReadVariableOp/batch_normalization_27/beta/Read/ReadVariableOp6batch_normalization_27/moving_mean/Read/ReadVariableOp:batch_normalization_27/moving_variance/Read/ReadVariableOp$conv2d_29/kernel/Read/ReadVariableOp"conv2d_29/bias/Read/ReadVariableOp$cond_1/Adam/iter/Read/ReadVariableOp&cond_1/Adam/beta_1/Read/ReadVariableOp&cond_1/Adam/beta_2/Read/ReadVariableOp%cond_1/Adam/decay/Read/ReadVariableOp-cond_1/Adam/learning_rate/Read/ReadVariableOp&current_loss_scale/Read/ReadVariableOpgood_steps/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp2cond_1/Adam/conv2d_16/kernel/m/Read/ReadVariableOp0cond_1/Adam/conv2d_16/bias/m/Read/ReadVariableOp>cond_1/Adam/batch_normalization_15/gamma/m/Read/ReadVariableOp=cond_1/Adam/batch_normalization_15/beta/m/Read/ReadVariableOp2cond_1/Adam/conv2d_18/kernel/m/Read/ReadVariableOp0cond_1/Adam/conv2d_18/bias/m/Read/ReadVariableOp>cond_1/Adam/batch_normalization_17/gamma/m/Read/ReadVariableOp=cond_1/Adam/batch_normalization_17/beta/m/Read/ReadVariableOp2cond_1/Adam/conv2d_20/kernel/m/Read/ReadVariableOp0cond_1/Adam/conv2d_20/bias/m/Read/ReadVariableOp>cond_1/Adam/batch_normalization_19/gamma/m/Read/ReadVariableOp=cond_1/Adam/batch_normalization_19/beta/m/Read/ReadVariableOp2cond_1/Adam/conv2d_22/kernel/m/Read/ReadVariableOp0cond_1/Adam/conv2d_22/bias/m/Read/ReadVariableOp>cond_1/Adam/batch_normalization_21/gamma/m/Read/ReadVariableOp=cond_1/Adam/batch_normalization_21/beta/m/Read/ReadVariableOp;cond_1/Adam/conv2d_transpose_3/kernel/m/Read/ReadVariableOp9cond_1/Adam/conv2d_transpose_3/bias/m/Read/ReadVariableOp2cond_1/Adam/conv2d_24/kernel/m/Read/ReadVariableOp0cond_1/Adam/conv2d_24/bias/m/Read/ReadVariableOp>cond_1/Adam/batch_normalization_23/gamma/m/Read/ReadVariableOp=cond_1/Adam/batch_normalization_23/beta/m/Read/ReadVariableOp;cond_1/Adam/conv2d_transpose_4/kernel/m/Read/ReadVariableOp9cond_1/Adam/conv2d_transpose_4/bias/m/Read/ReadVariableOp2cond_1/Adam/conv2d_26/kernel/m/Read/ReadVariableOp0cond_1/Adam/conv2d_26/bias/m/Read/ReadVariableOp>cond_1/Adam/batch_normalization_25/gamma/m/Read/ReadVariableOp=cond_1/Adam/batch_normalization_25/beta/m/Read/ReadVariableOp;cond_1/Adam/conv2d_transpose_5/kernel/m/Read/ReadVariableOp9cond_1/Adam/conv2d_transpose_5/bias/m/Read/ReadVariableOp2cond_1/Adam/conv2d_28/kernel/m/Read/ReadVariableOp0cond_1/Adam/conv2d_28/bias/m/Read/ReadVariableOp>cond_1/Adam/batch_normalization_27/gamma/m/Read/ReadVariableOp=cond_1/Adam/batch_normalization_27/beta/m/Read/ReadVariableOp2cond_1/Adam/conv2d_29/kernel/m/Read/ReadVariableOp0cond_1/Adam/conv2d_29/bias/m/Read/ReadVariableOp2cond_1/Adam/conv2d_16/kernel/v/Read/ReadVariableOp0cond_1/Adam/conv2d_16/bias/v/Read/ReadVariableOp>cond_1/Adam/batch_normalization_15/gamma/v/Read/ReadVariableOp=cond_1/Adam/batch_normalization_15/beta/v/Read/ReadVariableOp2cond_1/Adam/conv2d_18/kernel/v/Read/ReadVariableOp0cond_1/Adam/conv2d_18/bias/v/Read/ReadVariableOp>cond_1/Adam/batch_normalization_17/gamma/v/Read/ReadVariableOp=cond_1/Adam/batch_normalization_17/beta/v/Read/ReadVariableOp2cond_1/Adam/conv2d_20/kernel/v/Read/ReadVariableOp0cond_1/Adam/conv2d_20/bias/v/Read/ReadVariableOp>cond_1/Adam/batch_normalization_19/gamma/v/Read/ReadVariableOp=cond_1/Adam/batch_normalization_19/beta/v/Read/ReadVariableOp2cond_1/Adam/conv2d_22/kernel/v/Read/ReadVariableOp0cond_1/Adam/conv2d_22/bias/v/Read/ReadVariableOp>cond_1/Adam/batch_normalization_21/gamma/v/Read/ReadVariableOp=cond_1/Adam/batch_normalization_21/beta/v/Read/ReadVariableOp;cond_1/Adam/conv2d_transpose_3/kernel/v/Read/ReadVariableOp9cond_1/Adam/conv2d_transpose_3/bias/v/Read/ReadVariableOp2cond_1/Adam/conv2d_24/kernel/v/Read/ReadVariableOp0cond_1/Adam/conv2d_24/bias/v/Read/ReadVariableOp>cond_1/Adam/batch_normalization_23/gamma/v/Read/ReadVariableOp=cond_1/Adam/batch_normalization_23/beta/v/Read/ReadVariableOp;cond_1/Adam/conv2d_transpose_4/kernel/v/Read/ReadVariableOp9cond_1/Adam/conv2d_transpose_4/bias/v/Read/ReadVariableOp2cond_1/Adam/conv2d_26/kernel/v/Read/ReadVariableOp0cond_1/Adam/conv2d_26/bias/v/Read/ReadVariableOp>cond_1/Adam/batch_normalization_25/gamma/v/Read/ReadVariableOp=cond_1/Adam/batch_normalization_25/beta/v/Read/ReadVariableOp;cond_1/Adam/conv2d_transpose_5/kernel/v/Read/ReadVariableOp9cond_1/Adam/conv2d_transpose_5/bias/v/Read/ReadVariableOp2cond_1/Adam/conv2d_28/kernel/v/Read/ReadVariableOp0cond_1/Adam/conv2d_28/bias/v/Read/ReadVariableOp>cond_1/Adam/batch_normalization_27/gamma/v/Read/ReadVariableOp=cond_1/Adam/batch_normalization_27/beta/v/Read/ReadVariableOp2cond_1/Adam/conv2d_29/kernel/v/Read/ReadVariableOp0cond_1/Adam/conv2d_29/bias/v/Read/ReadVariableOpConst*?
Tin?
?2?		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *'
f"R 
__inference__traced_save_42793
?$
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_16/kernelconv2d_16/biasbatch_normalization_15/gammabatch_normalization_15/beta"batch_normalization_15/moving_mean&batch_normalization_15/moving_varianceconv2d_18/kernelconv2d_18/biasbatch_normalization_17/gammabatch_normalization_17/beta"batch_normalization_17/moving_mean&batch_normalization_17/moving_varianceconv2d_20/kernelconv2d_20/biasbatch_normalization_19/gammabatch_normalization_19/beta"batch_normalization_19/moving_mean&batch_normalization_19/moving_varianceconv2d_22/kernelconv2d_22/biasbatch_normalization_21/gammabatch_normalization_21/beta"batch_normalization_21/moving_mean&batch_normalization_21/moving_varianceconv2d_transpose_3/kernelconv2d_transpose_3/biasconv2d_24/kernelconv2d_24/biasbatch_normalization_23/gammabatch_normalization_23/beta"batch_normalization_23/moving_mean&batch_normalization_23/moving_varianceconv2d_transpose_4/kernelconv2d_transpose_4/biasconv2d_26/kernelconv2d_26/biasbatch_normalization_25/gammabatch_normalization_25/beta"batch_normalization_25/moving_mean&batch_normalization_25/moving_varianceconv2d_transpose_5/kernelconv2d_transpose_5/biasconv2d_28/kernelconv2d_28/biasbatch_normalization_27/gammabatch_normalization_27/beta"batch_normalization_27/moving_mean&batch_normalization_27/moving_varianceconv2d_29/kernelconv2d_29/biascond_1/Adam/itercond_1/Adam/beta_1cond_1/Adam/beta_2cond_1/Adam/decaycond_1/Adam/learning_ratecurrent_loss_scale
good_stepstotalcounttotal_1count_1cond_1/Adam/conv2d_16/kernel/mcond_1/Adam/conv2d_16/bias/m*cond_1/Adam/batch_normalization_15/gamma/m)cond_1/Adam/batch_normalization_15/beta/mcond_1/Adam/conv2d_18/kernel/mcond_1/Adam/conv2d_18/bias/m*cond_1/Adam/batch_normalization_17/gamma/m)cond_1/Adam/batch_normalization_17/beta/mcond_1/Adam/conv2d_20/kernel/mcond_1/Adam/conv2d_20/bias/m*cond_1/Adam/batch_normalization_19/gamma/m)cond_1/Adam/batch_normalization_19/beta/mcond_1/Adam/conv2d_22/kernel/mcond_1/Adam/conv2d_22/bias/m*cond_1/Adam/batch_normalization_21/gamma/m)cond_1/Adam/batch_normalization_21/beta/m'cond_1/Adam/conv2d_transpose_3/kernel/m%cond_1/Adam/conv2d_transpose_3/bias/mcond_1/Adam/conv2d_24/kernel/mcond_1/Adam/conv2d_24/bias/m*cond_1/Adam/batch_normalization_23/gamma/m)cond_1/Adam/batch_normalization_23/beta/m'cond_1/Adam/conv2d_transpose_4/kernel/m%cond_1/Adam/conv2d_transpose_4/bias/mcond_1/Adam/conv2d_26/kernel/mcond_1/Adam/conv2d_26/bias/m*cond_1/Adam/batch_normalization_25/gamma/m)cond_1/Adam/batch_normalization_25/beta/m'cond_1/Adam/conv2d_transpose_5/kernel/m%cond_1/Adam/conv2d_transpose_5/bias/mcond_1/Adam/conv2d_28/kernel/mcond_1/Adam/conv2d_28/bias/m*cond_1/Adam/batch_normalization_27/gamma/m)cond_1/Adam/batch_normalization_27/beta/mcond_1/Adam/conv2d_29/kernel/mcond_1/Adam/conv2d_29/bias/mcond_1/Adam/conv2d_16/kernel/vcond_1/Adam/conv2d_16/bias/v*cond_1/Adam/batch_normalization_15/gamma/v)cond_1/Adam/batch_normalization_15/beta/vcond_1/Adam/conv2d_18/kernel/vcond_1/Adam/conv2d_18/bias/v*cond_1/Adam/batch_normalization_17/gamma/v)cond_1/Adam/batch_normalization_17/beta/vcond_1/Adam/conv2d_20/kernel/vcond_1/Adam/conv2d_20/bias/v*cond_1/Adam/batch_normalization_19/gamma/v)cond_1/Adam/batch_normalization_19/beta/vcond_1/Adam/conv2d_22/kernel/vcond_1/Adam/conv2d_22/bias/v*cond_1/Adam/batch_normalization_21/gamma/v)cond_1/Adam/batch_normalization_21/beta/v'cond_1/Adam/conv2d_transpose_3/kernel/v%cond_1/Adam/conv2d_transpose_3/bias/vcond_1/Adam/conv2d_24/kernel/vcond_1/Adam/conv2d_24/bias/v*cond_1/Adam/batch_normalization_23/gamma/v)cond_1/Adam/batch_normalization_23/beta/v'cond_1/Adam/conv2d_transpose_4/kernel/v%cond_1/Adam/conv2d_transpose_4/bias/vcond_1/Adam/conv2d_26/kernel/vcond_1/Adam/conv2d_26/bias/v*cond_1/Adam/batch_normalization_25/gamma/v)cond_1/Adam/batch_normalization_25/beta/v'cond_1/Adam/conv2d_transpose_5/kernel/v%cond_1/Adam/conv2d_transpose_5/bias/vcond_1/Adam/conv2d_28/kernel/vcond_1/Adam/conv2d_28/bias/v*cond_1/Adam/batch_normalization_27/gamma/v)cond_1/Adam/batch_normalization_27/beta/vcond_1/Adam/conv2d_29/kernel/vcond_1/Adam/conv2d_29/bias/v*?
Tin?
?2?*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? **
f%R#
!__inference__traced_restore_43202??'
?
?
Q__inference_batch_normalization_23_layer_call_and_return_conditional_losses_38121

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
??
?
B__inference_model_1_layer_call_and_return_conditional_losses_39653
input_2
conv2d_16_39515
conv2d_16_39517 
batch_normalization_15_39520 
batch_normalization_15_39522 
batch_normalization_15_39524 
batch_normalization_15_39526
conv2d_18_39532
conv2d_18_39534 
batch_normalization_17_39537 
batch_normalization_17_39539 
batch_normalization_17_39541 
batch_normalization_17_39543
conv2d_20_39549
conv2d_20_39551 
batch_normalization_19_39554 
batch_normalization_19_39556 
batch_normalization_19_39558 
batch_normalization_19_39560
conv2d_22_39566
conv2d_22_39568 
batch_normalization_21_39571 
batch_normalization_21_39573 
batch_normalization_21_39575 
batch_normalization_21_39577
conv2d_transpose_3_39581
conv2d_transpose_3_39583
conv2d_24_39588
conv2d_24_39590 
batch_normalization_23_39593 
batch_normalization_23_39595 
batch_normalization_23_39597 
batch_normalization_23_39599
conv2d_transpose_4_39603
conv2d_transpose_4_39605
conv2d_26_39610
conv2d_26_39612 
batch_normalization_25_39615 
batch_normalization_25_39617 
batch_normalization_25_39619 
batch_normalization_25_39621
conv2d_transpose_5_39625
conv2d_transpose_5_39627
conv2d_28_39632
conv2d_28_39634 
batch_normalization_27_39637 
batch_normalization_27_39639 
batch_normalization_27_39641 
batch_normalization_27_39643
conv2d_29_39647
conv2d_29_39649
identity??.batch_normalization_15/StatefulPartitionedCall?.batch_normalization_17/StatefulPartitionedCall?.batch_normalization_19/StatefulPartitionedCall?.batch_normalization_21/StatefulPartitionedCall?.batch_normalization_23/StatefulPartitionedCall?.batch_normalization_25/StatefulPartitionedCall?.batch_normalization_27/StatefulPartitionedCall?!conv2d_16/StatefulPartitionedCall?!conv2d_18/StatefulPartitionedCall?!conv2d_20/StatefulPartitionedCall?!conv2d_22/StatefulPartitionedCall?!conv2d_24/StatefulPartitionedCall?!conv2d_26/StatefulPartitionedCall?!conv2d_28/StatefulPartitionedCall?!conv2d_29/StatefulPartitionedCall?*conv2d_transpose_3/StatefulPartitionedCall?*conv2d_transpose_4/StatefulPartitionedCall?*conv2d_transpose_5/StatefulPartitionedCallh
CastCastinput_2*

DstT0*

SrcT0*1
_output_shapes
:???????????2
Cast?
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCallCast:y:0conv2d_16_39515conv2d_16_39517*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_conv2d_16_layer_call_and_return_conditional_losses_384492#
!conv2d_16/StatefulPartitionedCall?
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0batch_normalization_15_39520batch_normalization_15_39522batch_normalization_15_39524batch_normalization_15_39526*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_3850220
.batch_normalization_15/StatefulPartitionedCall?
activation_15/PartitionedCallPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *Q
fLRJ
H__inference_activation_15_layer_call_and_return_conditional_losses_385432
activation_15/PartitionedCall?
max_pooling2d_3/PartitionedCallPartitionedCall&activation_15/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_376402!
max_pooling2d_3/PartitionedCall?
dropout_6/PartitionedCallPartitionedCall(max_pooling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_385692
dropout_6/PartitionedCall?
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCall"dropout_6/PartitionedCall:output:0conv2d_18_39532conv2d_18_39534*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_conv2d_18_layer_call_and_return_conditional_losses_385942#
!conv2d_18/StatefulPartitionedCall?
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0batch_normalization_17_39537batch_normalization_17_39539batch_normalization_17_39541batch_normalization_17_39543*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_3864720
.batch_normalization_17/StatefulPartitionedCall?
activation_17/PartitionedCallPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *Q
fLRJ
H__inference_activation_17_layer_call_and_return_conditional_losses_386882
activation_17/PartitionedCall?
max_pooling2d_4/PartitionedCallPartitionedCall&activation_17/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_377562!
max_pooling2d_4/PartitionedCall?
dropout_7/PartitionedCallPartitionedCall(max_pooling2d_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_387142
dropout_7/PartitionedCall?
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall"dropout_7/PartitionedCall:output:0conv2d_20_39549conv2d_20_39551*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_conv2d_20_layer_call_and_return_conditional_losses_387392#
!conv2d_20/StatefulPartitionedCall?
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0batch_normalization_19_39554batch_normalization_19_39556batch_normalization_19_39558batch_normalization_19_39560*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_3879220
.batch_normalization_19/StatefulPartitionedCall?
activation_19/PartitionedCallPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *Q
fLRJ
H__inference_activation_19_layer_call_and_return_conditional_losses_388332
activation_19/PartitionedCall?
max_pooling2d_5/PartitionedCallPartitionedCall&activation_19/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????D\?* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *S
fNRL
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_378722!
max_pooling2d_5/PartitionedCall?
dropout_8/PartitionedCallPartitionedCall(max_pooling2d_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????D\?* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_388592
dropout_8/PartitionedCall?
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall"dropout_8/PartitionedCall:output:0conv2d_22_39566conv2d_22_39568*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????D\?*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_conv2d_22_layer_call_and_return_conditional_losses_388842#
!conv2d_22/StatefulPartitionedCall?
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0batch_normalization_21_39571batch_normalization_21_39573batch_normalization_21_39575batch_normalization_21_39577*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????D\?*&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_21_layer_call_and_return_conditional_losses_3893720
.batch_normalization_21/StatefulPartitionedCall?
activation_21/PartitionedCallPartitionedCall7batch_normalization_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????D\?* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *Q
fLRJ
H__inference_activation_21_layer_call_and_return_conditional_losses_389782
activation_21/PartitionedCall?
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall&activation_21/PartitionedCall:output:0conv2d_transpose_3_39581conv2d_transpose_3_39583*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *V
fQRO
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_380182,
*conv2d_transpose_3/StatefulPartitionedCall?
concatenate_3/PartitionedCallPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0&activation_19/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *Q
fLRJ
H__inference_concatenate_3_layer_call_and_return_conditional_losses_389982
concatenate_3/PartitionedCall?
dropout_9/PartitionedCallPartitionedCall&concatenate_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_dropout_9_layer_call_and_return_conditional_losses_390242
dropout_9/PartitionedCall?
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall"dropout_9/PartitionedCall:output:0conv2d_24_39588conv2d_24_39590*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_conv2d_24_layer_call_and_return_conditional_losses_390492#
!conv2d_24/StatefulPartitionedCall?
.batch_normalization_23/StatefulPartitionedCallStatefulPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0batch_normalization_23_39593batch_normalization_23_39595batch_normalization_23_39597batch_normalization_23_39599*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_23_layer_call_and_return_conditional_losses_3910220
.batch_normalization_23/StatefulPartitionedCall?
activation_23/PartitionedCallPartitionedCall7batch_normalization_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *Q
fLRJ
H__inference_activation_23_layer_call_and_return_conditional_losses_391432
activation_23/PartitionedCall?
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall&activation_23/PartitionedCall:output:0conv2d_transpose_4_39603conv2d_transpose_4_39605*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *V
fQRO
M__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_381682,
*conv2d_transpose_4/StatefulPartitionedCall?
concatenate_4/PartitionedCallPartitionedCall3conv2d_transpose_4/StatefulPartitionedCall:output:0&activation_17/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *Q
fLRJ
H__inference_concatenate_4_layer_call_and_return_conditional_losses_391632
concatenate_4/PartitionedCall?
dropout_10/PartitionedCallPartitionedCall&concatenate_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *N
fIRG
E__inference_dropout_10_layer_call_and_return_conditional_losses_391892
dropout_10/PartitionedCall?
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall#dropout_10/PartitionedCall:output:0conv2d_26_39610conv2d_26_39612*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_conv2d_26_layer_call_and_return_conditional_losses_392142#
!conv2d_26/StatefulPartitionedCall?
.batch_normalization_25/StatefulPartitionedCallStatefulPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0batch_normalization_25_39615batch_normalization_25_39617batch_normalization_25_39619batch_normalization_25_39621*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_25_layer_call_and_return_conditional_losses_3926720
.batch_normalization_25/StatefulPartitionedCall?
activation_25/PartitionedCallPartitionedCall7batch_normalization_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *Q
fLRJ
H__inference_activation_25_layer_call_and_return_conditional_losses_393082
activation_25/PartitionedCall?
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall&activation_25/PartitionedCall:output:0conv2d_transpose_5_39625conv2d_transpose_5_39627*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *V
fQRO
M__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_383182,
*conv2d_transpose_5/StatefulPartitionedCall?
concatenate_5/PartitionedCallPartitionedCall3conv2d_transpose_5/StatefulPartitionedCall:output:0&activation_15/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *Q
fLRJ
H__inference_concatenate_5_layer_call_and_return_conditional_losses_393282
concatenate_5/PartitionedCall?
dropout_11/PartitionedCallPartitionedCall&concatenate_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *N
fIRG
E__inference_dropout_11_layer_call_and_return_conditional_losses_393542
dropout_11/PartitionedCall?
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCall#dropout_11/PartitionedCall:output:0conv2d_28_39632conv2d_28_39634*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_conv2d_28_layer_call_and_return_conditional_losses_393792#
!conv2d_28/StatefulPartitionedCall?
.batch_normalization_27/StatefulPartitionedCallStatefulPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0batch_normalization_27_39637batch_normalization_27_39639batch_normalization_27_39641batch_normalization_27_39643*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_27_layer_call_and_return_conditional_losses_3943220
.batch_normalization_27/StatefulPartitionedCall?
activation_27/PartitionedCallPartitionedCall7batch_normalization_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *Q
fLRJ
H__inference_activation_27_layer_call_and_return_conditional_losses_394732
activation_27/PartitionedCall?
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall&activation_27/PartitionedCall:output:0conv2d_29_39647conv2d_29_39649*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_conv2d_29_layer_call_and_return_conditional_losses_394942#
!conv2d_29/StatefulPartitionedCall?
IdentityIdentity*conv2d_29/StatefulPartitionedCall:output:0/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall/^batch_normalization_19/StatefulPartitionedCall/^batch_normalization_21/StatefulPartitionedCall/^batch_normalization_23/StatefulPartitionedCall/^batch_normalization_25/StatefulPartitionedCall/^batch_normalization_27/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall"^conv2d_22/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall"^conv2d_28/StatefulPartitionedCall"^conv2d_29/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall+^conv2d_transpose_4/StatefulPartitionedCall+^conv2d_transpose_5/StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2`
.batch_normalization_23/StatefulPartitionedCall.batch_normalization_23/StatefulPartitionedCall2`
.batch_normalization_25/StatefulPartitionedCall.batch_normalization_25/StatefulPartitionedCall2`
.batch_normalization_27/StatefulPartitionedCall.batch_normalization_27/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_2
?
c
*__inference_dropout_11_layer_call_fn_42185

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *N
fIRG
E__inference_dropout_11_layer_call_and_return_conditional_losses_393492
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????@22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_27_layer_call_and_return_conditional_losses_42295

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_25_layer_call_fn_42076

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_25_layer_call_and_return_conditional_losses_382712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_27_layer_call_and_return_conditional_losses_38390

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_27_layer_call_fn_42275

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_27_layer_call_and_return_conditional_losses_394322
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:??????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_40263
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./012*8
config_proto(&

CPU

GPU2*0J

  ?E8? *)
f$R"
 __inference__wrapped_model_375302
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_2
?
?
D__inference_conv2d_24_layer_call_and_return_conditional_losses_39049

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*(
_output_shapes
:??2
Conv2D/Cast?
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:?2
BiasAdd/Cast}
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*2
_output_shapes 
:????????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
t
H__inference_concatenate_3_layer_call_and_return_conditional_losses_41759
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*2
_output_shapes 
:????????????2
concatn
IdentityIdentityconcat:output:0*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:,????????????????????????????:????????????:l h
B
_output_shapes0
.:,????????????????????????????
"
_user_specified_name
inputs/0:\X
2
_output_shapes 
:????????????
"
_user_specified_name
inputs/1
?
~
)__inference_conv2d_20_layer_call_fn_41428

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_conv2d_20_layer_call_and_return_conditional_losses_387392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????@::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_38502

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:??????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
2__inference_conv2d_transpose_4_layer_call_fn_38178

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *V
fQRO
M__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_381682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_25_layer_call_and_return_conditional_losses_38240

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
~
)__inference_conv2d_26_layer_call_fn_42012

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_conv2d_26_layer_call_and_return_conditional_losses_392142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:????????????::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
'__inference_model_1_layer_call_fn_40148
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./012*8
config_proto(&

CPU

GPU2*0J

  ?E8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_400452
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_2
?
c
D__inference_dropout_9_layer_call_and_return_conditional_losses_41777

inputs
identity?a
dropout/ConstConst*
_output_shapes
: *
dtype0*
value
B j?x2
dropout/Const~
dropout/MulMulinputsdropout/Const:output:0*
T0*2
_output_shapes 
:????????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*2
_output_shapes 
:????????????*
dtype02&
$dropout/random_uniform/RandomUniforms
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value
B j?\2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*2
_output_shapes 
:????????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*2
_output_shapes 
:????????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*2
_output_shapes 
:????????????2
dropout/Mul_1p
IdentityIdentitydropout/Mul_1:z:0*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*1
_input_shapes 
:????????????:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
D__inference_conv2d_26_layer_call_and_return_conditional_losses_42003

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02
Conv2D/ReadVariableOp?
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*'
_output_shapes
:?@2
Conv2D/Cast?
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpx
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@2
BiasAdd/Cast|
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*1
_output_shapes
:???????????@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_37592

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_23_layer_call_and_return_conditional_losses_41833

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
b
D__inference_dropout_7_layer_call_and_return_conditional_losses_38714

inputs

identity_1d
IdentityIdentityinputs*
T0*1
_output_shapes
:???????????@2

Identitys

Identity_1IdentityIdentity:output:0*
T0*1
_output_shapes
:???????????@2

Identity_1"!

identity_1Identity_1:output:0*0
_input_shapes
:???????????@:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_25_layer_call_fn_42140

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_25_layer_call_and_return_conditional_losses_392672
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
~
)__inference_conv2d_18_layer_call_fn_41242

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_conv2d_18_layer_call_and_return_conditional_losses_385942
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:??????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_41344

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_25_layer_call_and_return_conditional_losses_42050

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
K
/__inference_max_pooling2d_3_layer_call_fn_37646

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_376402
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
D__inference_conv2d_20_layer_call_and_return_conditional_losses_38739

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*'
_output_shapes
:@?2
Conv2D/Cast?
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:?2
BiasAdd/Cast}
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*2
_output_shapes 
:????????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_41140

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:??????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_37708

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
b
D__inference_dropout_6_layer_call_and_return_conditional_losses_38569

inputs

identity_1d
IdentityIdentityinputs*
T0*1
_output_shapes
:??????????? 2

Identitys

Identity_1IdentityIdentity:output:0*
T0*1
_output_shapes
:??????????? 2

Identity_1"!

identity_1Identity_1:output:0*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_41448

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
c
*__inference_dropout_10_layer_call_fn_41986

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *N
fIRG
E__inference_dropout_10_layer_call_and_return_conditional_losses_391842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*1
_input_shapes 
:????????????22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
׺
?
B__inference_model_1_layer_call_and_return_conditional_losses_39798

inputs
conv2d_16_39660
conv2d_16_39662 
batch_normalization_15_39665 
batch_normalization_15_39667 
batch_normalization_15_39669 
batch_normalization_15_39671
conv2d_18_39677
conv2d_18_39679 
batch_normalization_17_39682 
batch_normalization_17_39684 
batch_normalization_17_39686 
batch_normalization_17_39688
conv2d_20_39694
conv2d_20_39696 
batch_normalization_19_39699 
batch_normalization_19_39701 
batch_normalization_19_39703 
batch_normalization_19_39705
conv2d_22_39711
conv2d_22_39713 
batch_normalization_21_39716 
batch_normalization_21_39718 
batch_normalization_21_39720 
batch_normalization_21_39722
conv2d_transpose_3_39726
conv2d_transpose_3_39728
conv2d_24_39733
conv2d_24_39735 
batch_normalization_23_39738 
batch_normalization_23_39740 
batch_normalization_23_39742 
batch_normalization_23_39744
conv2d_transpose_4_39748
conv2d_transpose_4_39750
conv2d_26_39755
conv2d_26_39757 
batch_normalization_25_39760 
batch_normalization_25_39762 
batch_normalization_25_39764 
batch_normalization_25_39766
conv2d_transpose_5_39770
conv2d_transpose_5_39772
conv2d_28_39777
conv2d_28_39779 
batch_normalization_27_39782 
batch_normalization_27_39784 
batch_normalization_27_39786 
batch_normalization_27_39788
conv2d_29_39792
conv2d_29_39794
identity??.batch_normalization_15/StatefulPartitionedCall?.batch_normalization_17/StatefulPartitionedCall?.batch_normalization_19/StatefulPartitionedCall?.batch_normalization_21/StatefulPartitionedCall?.batch_normalization_23/StatefulPartitionedCall?.batch_normalization_25/StatefulPartitionedCall?.batch_normalization_27/StatefulPartitionedCall?!conv2d_16/StatefulPartitionedCall?!conv2d_18/StatefulPartitionedCall?!conv2d_20/StatefulPartitionedCall?!conv2d_22/StatefulPartitionedCall?!conv2d_24/StatefulPartitionedCall?!conv2d_26/StatefulPartitionedCall?!conv2d_28/StatefulPartitionedCall?!conv2d_29/StatefulPartitionedCall?*conv2d_transpose_3/StatefulPartitionedCall?*conv2d_transpose_4/StatefulPartitionedCall?*conv2d_transpose_5/StatefulPartitionedCall?"dropout_10/StatefulPartitionedCall?"dropout_11/StatefulPartitionedCall?!dropout_6/StatefulPartitionedCall?!dropout_7/StatefulPartitionedCall?!dropout_8/StatefulPartitionedCall?!dropout_9/StatefulPartitionedCallg
CastCastinputs*

DstT0*

SrcT0*1
_output_shapes
:???????????2
Cast?
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCallCast:y:0conv2d_16_39660conv2d_16_39662*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_conv2d_16_layer_call_and_return_conditional_losses_384492#
!conv2d_16/StatefulPartitionedCall?
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0batch_normalization_15_39665batch_normalization_15_39667batch_normalization_15_39669batch_normalization_15_39671*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_3848420
.batch_normalization_15/StatefulPartitionedCall?
activation_15/PartitionedCallPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *Q
fLRJ
H__inference_activation_15_layer_call_and_return_conditional_losses_385432
activation_15/PartitionedCall?
max_pooling2d_3/PartitionedCallPartitionedCall&activation_15/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_376402!
max_pooling2d_3/PartitionedCall?
!dropout_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_385642#
!dropout_6/StatefulPartitionedCall?
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCall*dropout_6/StatefulPartitionedCall:output:0conv2d_18_39677conv2d_18_39679*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_conv2d_18_layer_call_and_return_conditional_losses_385942#
!conv2d_18/StatefulPartitionedCall?
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0batch_normalization_17_39682batch_normalization_17_39684batch_normalization_17_39686batch_normalization_17_39688*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_3862920
.batch_normalization_17/StatefulPartitionedCall?
activation_17/PartitionedCallPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *Q
fLRJ
H__inference_activation_17_layer_call_and_return_conditional_losses_386882
activation_17/PartitionedCall?
max_pooling2d_4/PartitionedCallPartitionedCall&activation_17/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_377562!
max_pooling2d_4/PartitionedCall?
!dropout_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0"^dropout_6/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_387092#
!dropout_7/StatefulPartitionedCall?
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall*dropout_7/StatefulPartitionedCall:output:0conv2d_20_39694conv2d_20_39696*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_conv2d_20_layer_call_and_return_conditional_losses_387392#
!conv2d_20/StatefulPartitionedCall?
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0batch_normalization_19_39699batch_normalization_19_39701batch_normalization_19_39703batch_normalization_19_39705*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_3877420
.batch_normalization_19/StatefulPartitionedCall?
activation_19/PartitionedCallPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *Q
fLRJ
H__inference_activation_19_layer_call_and_return_conditional_losses_388332
activation_19/PartitionedCall?
max_pooling2d_5/PartitionedCallPartitionedCall&activation_19/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????D\?* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *S
fNRL
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_378722!
max_pooling2d_5/PartitionedCall?
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0"^dropout_7/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????D\?* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_388542#
!dropout_8/StatefulPartitionedCall?
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall*dropout_8/StatefulPartitionedCall:output:0conv2d_22_39711conv2d_22_39713*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????D\?*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_conv2d_22_layer_call_and_return_conditional_losses_388842#
!conv2d_22/StatefulPartitionedCall?
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0batch_normalization_21_39716batch_normalization_21_39718batch_normalization_21_39720batch_normalization_21_39722*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????D\?*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_21_layer_call_and_return_conditional_losses_3891920
.batch_normalization_21/StatefulPartitionedCall?
activation_21/PartitionedCallPartitionedCall7batch_normalization_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????D\?* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *Q
fLRJ
H__inference_activation_21_layer_call_and_return_conditional_losses_389782
activation_21/PartitionedCall?
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall&activation_21/PartitionedCall:output:0conv2d_transpose_3_39726conv2d_transpose_3_39728*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *V
fQRO
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_380182,
*conv2d_transpose_3/StatefulPartitionedCall?
concatenate_3/PartitionedCallPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0&activation_19/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *Q
fLRJ
H__inference_concatenate_3_layer_call_and_return_conditional_losses_389982
concatenate_3/PartitionedCall?
!dropout_9/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0"^dropout_8/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_dropout_9_layer_call_and_return_conditional_losses_390192#
!dropout_9/StatefulPartitionedCall?
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall*dropout_9/StatefulPartitionedCall:output:0conv2d_24_39733conv2d_24_39735*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_conv2d_24_layer_call_and_return_conditional_losses_390492#
!conv2d_24/StatefulPartitionedCall?
.batch_normalization_23/StatefulPartitionedCallStatefulPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0batch_normalization_23_39738batch_normalization_23_39740batch_normalization_23_39742batch_normalization_23_39744*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_23_layer_call_and_return_conditional_losses_3908420
.batch_normalization_23/StatefulPartitionedCall?
activation_23/PartitionedCallPartitionedCall7batch_normalization_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *Q
fLRJ
H__inference_activation_23_layer_call_and_return_conditional_losses_391432
activation_23/PartitionedCall?
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall&activation_23/PartitionedCall:output:0conv2d_transpose_4_39748conv2d_transpose_4_39750*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *V
fQRO
M__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_381682,
*conv2d_transpose_4/StatefulPartitionedCall?
concatenate_4/PartitionedCallPartitionedCall3conv2d_transpose_4/StatefulPartitionedCall:output:0&activation_17/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *Q
fLRJ
H__inference_concatenate_4_layer_call_and_return_conditional_losses_391632
concatenate_4/PartitionedCall?
"dropout_10/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0"^dropout_9/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *N
fIRG
E__inference_dropout_10_layer_call_and_return_conditional_losses_391842$
"dropout_10/StatefulPartitionedCall?
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall+dropout_10/StatefulPartitionedCall:output:0conv2d_26_39755conv2d_26_39757*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_conv2d_26_layer_call_and_return_conditional_losses_392142#
!conv2d_26/StatefulPartitionedCall?
.batch_normalization_25/StatefulPartitionedCallStatefulPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0batch_normalization_25_39760batch_normalization_25_39762batch_normalization_25_39764batch_normalization_25_39766*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_25_layer_call_and_return_conditional_losses_3924920
.batch_normalization_25/StatefulPartitionedCall?
activation_25/PartitionedCallPartitionedCall7batch_normalization_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *Q
fLRJ
H__inference_activation_25_layer_call_and_return_conditional_losses_393082
activation_25/PartitionedCall?
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall&activation_25/PartitionedCall:output:0conv2d_transpose_5_39770conv2d_transpose_5_39772*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *V
fQRO
M__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_383182,
*conv2d_transpose_5/StatefulPartitionedCall?
concatenate_5/PartitionedCallPartitionedCall3conv2d_transpose_5/StatefulPartitionedCall:output:0&activation_15/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *Q
fLRJ
H__inference_concatenate_5_layer_call_and_return_conditional_losses_393282
concatenate_5/PartitionedCall?
"dropout_11/StatefulPartitionedCallStatefulPartitionedCall&concatenate_5/PartitionedCall:output:0#^dropout_10/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *N
fIRG
E__inference_dropout_11_layer_call_and_return_conditional_losses_393492$
"dropout_11/StatefulPartitionedCall?
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCall+dropout_11/StatefulPartitionedCall:output:0conv2d_28_39777conv2d_28_39779*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_conv2d_28_layer_call_and_return_conditional_losses_393792#
!conv2d_28/StatefulPartitionedCall?
.batch_normalization_27/StatefulPartitionedCallStatefulPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0batch_normalization_27_39782batch_normalization_27_39784batch_normalization_27_39786batch_normalization_27_39788*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_27_layer_call_and_return_conditional_losses_3941420
.batch_normalization_27/StatefulPartitionedCall?
activation_27/PartitionedCallPartitionedCall7batch_normalization_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *Q
fLRJ
H__inference_activation_27_layer_call_and_return_conditional_losses_394732
activation_27/PartitionedCall?
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall&activation_27/PartitionedCall:output:0conv2d_29_39792conv2d_29_39794*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_conv2d_29_layer_call_and_return_conditional_losses_394942#
!conv2d_29/StatefulPartitionedCall?
IdentityIdentity*conv2d_29/StatefulPartitionedCall:output:0/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall/^batch_normalization_19/StatefulPartitionedCall/^batch_normalization_21/StatefulPartitionedCall/^batch_normalization_23/StatefulPartitionedCall/^batch_normalization_25/StatefulPartitionedCall/^batch_normalization_27/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall"^conv2d_22/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall"^conv2d_28/StatefulPartitionedCall"^conv2d_29/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall+^conv2d_transpose_4/StatefulPartitionedCall+^conv2d_transpose_5/StatefulPartitionedCall#^dropout_10/StatefulPartitionedCall#^dropout_11/StatefulPartitionedCall"^dropout_6/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall"^dropout_9/StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2`
.batch_normalization_23/StatefulPartitionedCall.batch_normalization_23/StatefulPartitionedCall2`
.batch_normalization_25/StatefulPartitionedCall.batch_normalization_25/StatefulPartitionedCall2`
.batch_normalization_27/StatefulPartitionedCall.batch_normalization_27/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall2H
"dropout_10/StatefulPartitionedCall"dropout_10/StatefulPartitionedCall2H
"dropout_11/StatefulPartitionedCall"dropout_11/StatefulPartitionedCall2F
!dropout_6/StatefulPartitionedCall!dropout_6/StatefulPartitionedCall2F
!dropout_7/StatefulPartitionedCall!dropout_7/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall2F
!dropout_9/StatefulPartitionedCall!dropout_9/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
D__inference_conv2d_18_layer_call_and_return_conditional_losses_41233

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: @2
Conv2D/Cast?
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpx
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@2
BiasAdd/Cast|
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*1
_output_shapes
:???????????@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:??????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_41076

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
I
-__inference_activation_19_layer_call_fn_41566

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *Q
fLRJ
H__inference_activation_19_layer_call_and_return_conditional_losses_388332
PartitionedCallw
IdentityIdentityPartitionedCall:output:0*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*1
_input_shapes 
:????????????:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
c
D__inference_dropout_7_layer_call_and_return_conditional_losses_38709

inputs
identity?a
dropout/ConstConst*
_output_shapes
: *
dtype0*
value
B j?x2
dropout/Const}
dropout/MulMulinputsdropout/Const:output:0*
T0*1
_output_shapes
:???????????@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*1
_output_shapes
:???????????@*
dtype02&
$dropout/random_uniform/RandomUniforms
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value
B j?\2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:???????????@2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:???????????@2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*1
_output_shapes
:???????????@2
dropout/Mul_1o
IdentityIdentitydropout/Mul_1:z:0*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????@:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
I
-__inference_activation_17_layer_call_fn_41380

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *Q
fLRJ
H__inference_activation_17_layer_call_and_return_conditional_losses_386882
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????@:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
~
)__inference_conv2d_22_layer_call_fn_41614

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????D\?*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_conv2d_22_layer_call_and_return_conditional_losses_388842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????D\?2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????D\?::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????D\?
 
_user_specified_nameinputs
?
c
D__inference_dropout_8_layer_call_and_return_conditional_losses_38854

inputs
identity?a
dropout/ConstConst*
_output_shapes
: *
dtype0*
value
B j?x2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:?????????D\?2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:?????????D\?*
dtype02&
$dropout/random_uniform/RandomUniforms
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value
B j?\2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:?????????D\?2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:?????????D\?2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:?????????D\?2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:?????????D\?2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????D\?:X T
0
_output_shapes
:?????????D\?
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_25_layer_call_and_return_conditional_losses_39249

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_21_layer_call_and_return_conditional_losses_41634

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_27_layer_call_and_return_conditional_losses_42231

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:??????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_37739

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
Y
-__inference_concatenate_5_layer_call_fn_42163
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *Q
fLRJ
H__inference_concatenate_5_layer_call_and_return_conditional_losses_393282
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:+??????????????????????????? :??????????? :k g
A
_output_shapes/
-:+??????????????????????????? 
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:??????????? 
"
_user_specified_name
inputs/1
?
~
)__inference_conv2d_16_layer_call_fn_41056

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_conv2d_16_layer_call_and_return_conditional_losses_384492
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_37872

inputs
identity?
MaxPoolMaxPoolinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
D__inference_conv2d_29_layer_call_and_return_conditional_losses_39494

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: 2
Conv2D/Cast?
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpx
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:2
BiasAdd/Cast|
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*1
_output_shapes
:???????????2	
BiasAddk
SigmoidSigmoidBiasAdd:output:0*
T0*1
_output_shapes
:???????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:??????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_27_layer_call_and_return_conditional_losses_39432

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:??????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
ں
?
B__inference_model_1_layer_call_and_return_conditional_losses_39511
input_2
conv2d_16_38460
conv2d_16_38462 
batch_normalization_15_38529 
batch_normalization_15_38531 
batch_normalization_15_38533 
batch_normalization_15_38535
conv2d_18_38605
conv2d_18_38607 
batch_normalization_17_38674 
batch_normalization_17_38676 
batch_normalization_17_38678 
batch_normalization_17_38680
conv2d_20_38750
conv2d_20_38752 
batch_normalization_19_38819 
batch_normalization_19_38821 
batch_normalization_19_38823 
batch_normalization_19_38825
conv2d_22_38895
conv2d_22_38897 
batch_normalization_21_38964 
batch_normalization_21_38966 
batch_normalization_21_38968 
batch_normalization_21_38970
conv2d_transpose_3_38986
conv2d_transpose_3_38988
conv2d_24_39060
conv2d_24_39062 
batch_normalization_23_39129 
batch_normalization_23_39131 
batch_normalization_23_39133 
batch_normalization_23_39135
conv2d_transpose_4_39151
conv2d_transpose_4_39153
conv2d_26_39225
conv2d_26_39227 
batch_normalization_25_39294 
batch_normalization_25_39296 
batch_normalization_25_39298 
batch_normalization_25_39300
conv2d_transpose_5_39316
conv2d_transpose_5_39318
conv2d_28_39390
conv2d_28_39392 
batch_normalization_27_39459 
batch_normalization_27_39461 
batch_normalization_27_39463 
batch_normalization_27_39465
conv2d_29_39505
conv2d_29_39507
identity??.batch_normalization_15/StatefulPartitionedCall?.batch_normalization_17/StatefulPartitionedCall?.batch_normalization_19/StatefulPartitionedCall?.batch_normalization_21/StatefulPartitionedCall?.batch_normalization_23/StatefulPartitionedCall?.batch_normalization_25/StatefulPartitionedCall?.batch_normalization_27/StatefulPartitionedCall?!conv2d_16/StatefulPartitionedCall?!conv2d_18/StatefulPartitionedCall?!conv2d_20/StatefulPartitionedCall?!conv2d_22/StatefulPartitionedCall?!conv2d_24/StatefulPartitionedCall?!conv2d_26/StatefulPartitionedCall?!conv2d_28/StatefulPartitionedCall?!conv2d_29/StatefulPartitionedCall?*conv2d_transpose_3/StatefulPartitionedCall?*conv2d_transpose_4/StatefulPartitionedCall?*conv2d_transpose_5/StatefulPartitionedCall?"dropout_10/StatefulPartitionedCall?"dropout_11/StatefulPartitionedCall?!dropout_6/StatefulPartitionedCall?!dropout_7/StatefulPartitionedCall?!dropout_8/StatefulPartitionedCall?!dropout_9/StatefulPartitionedCallh
CastCastinput_2*

DstT0*

SrcT0*1
_output_shapes
:???????????2
Cast?
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCallCast:y:0conv2d_16_38460conv2d_16_38462*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_conv2d_16_layer_call_and_return_conditional_losses_384492#
!conv2d_16/StatefulPartitionedCall?
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0batch_normalization_15_38529batch_normalization_15_38531batch_normalization_15_38533batch_normalization_15_38535*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_3848420
.batch_normalization_15/StatefulPartitionedCall?
activation_15/PartitionedCallPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *Q
fLRJ
H__inference_activation_15_layer_call_and_return_conditional_losses_385432
activation_15/PartitionedCall?
max_pooling2d_3/PartitionedCallPartitionedCall&activation_15/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_376402!
max_pooling2d_3/PartitionedCall?
!dropout_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_385642#
!dropout_6/StatefulPartitionedCall?
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCall*dropout_6/StatefulPartitionedCall:output:0conv2d_18_38605conv2d_18_38607*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_conv2d_18_layer_call_and_return_conditional_losses_385942#
!conv2d_18/StatefulPartitionedCall?
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0batch_normalization_17_38674batch_normalization_17_38676batch_normalization_17_38678batch_normalization_17_38680*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_3862920
.batch_normalization_17/StatefulPartitionedCall?
activation_17/PartitionedCallPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *Q
fLRJ
H__inference_activation_17_layer_call_and_return_conditional_losses_386882
activation_17/PartitionedCall?
max_pooling2d_4/PartitionedCallPartitionedCall&activation_17/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_377562!
max_pooling2d_4/PartitionedCall?
!dropout_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0"^dropout_6/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_387092#
!dropout_7/StatefulPartitionedCall?
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall*dropout_7/StatefulPartitionedCall:output:0conv2d_20_38750conv2d_20_38752*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_conv2d_20_layer_call_and_return_conditional_losses_387392#
!conv2d_20/StatefulPartitionedCall?
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0batch_normalization_19_38819batch_normalization_19_38821batch_normalization_19_38823batch_normalization_19_38825*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_3877420
.batch_normalization_19/StatefulPartitionedCall?
activation_19/PartitionedCallPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *Q
fLRJ
H__inference_activation_19_layer_call_and_return_conditional_losses_388332
activation_19/PartitionedCall?
max_pooling2d_5/PartitionedCallPartitionedCall&activation_19/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????D\?* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *S
fNRL
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_378722!
max_pooling2d_5/PartitionedCall?
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0"^dropout_7/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????D\?* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_388542#
!dropout_8/StatefulPartitionedCall?
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall*dropout_8/StatefulPartitionedCall:output:0conv2d_22_38895conv2d_22_38897*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????D\?*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_conv2d_22_layer_call_and_return_conditional_losses_388842#
!conv2d_22/StatefulPartitionedCall?
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0batch_normalization_21_38964batch_normalization_21_38966batch_normalization_21_38968batch_normalization_21_38970*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????D\?*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_21_layer_call_and_return_conditional_losses_3891920
.batch_normalization_21/StatefulPartitionedCall?
activation_21/PartitionedCallPartitionedCall7batch_normalization_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????D\?* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *Q
fLRJ
H__inference_activation_21_layer_call_and_return_conditional_losses_389782
activation_21/PartitionedCall?
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall&activation_21/PartitionedCall:output:0conv2d_transpose_3_38986conv2d_transpose_3_38988*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *V
fQRO
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_380182,
*conv2d_transpose_3/StatefulPartitionedCall?
concatenate_3/PartitionedCallPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0&activation_19/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *Q
fLRJ
H__inference_concatenate_3_layer_call_and_return_conditional_losses_389982
concatenate_3/PartitionedCall?
!dropout_9/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0"^dropout_8/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_dropout_9_layer_call_and_return_conditional_losses_390192#
!dropout_9/StatefulPartitionedCall?
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall*dropout_9/StatefulPartitionedCall:output:0conv2d_24_39060conv2d_24_39062*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_conv2d_24_layer_call_and_return_conditional_losses_390492#
!conv2d_24/StatefulPartitionedCall?
.batch_normalization_23/StatefulPartitionedCallStatefulPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0batch_normalization_23_39129batch_normalization_23_39131batch_normalization_23_39133batch_normalization_23_39135*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_23_layer_call_and_return_conditional_losses_3908420
.batch_normalization_23/StatefulPartitionedCall?
activation_23/PartitionedCallPartitionedCall7batch_normalization_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *Q
fLRJ
H__inference_activation_23_layer_call_and_return_conditional_losses_391432
activation_23/PartitionedCall?
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall&activation_23/PartitionedCall:output:0conv2d_transpose_4_39151conv2d_transpose_4_39153*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *V
fQRO
M__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_381682,
*conv2d_transpose_4/StatefulPartitionedCall?
concatenate_4/PartitionedCallPartitionedCall3conv2d_transpose_4/StatefulPartitionedCall:output:0&activation_17/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *Q
fLRJ
H__inference_concatenate_4_layer_call_and_return_conditional_losses_391632
concatenate_4/PartitionedCall?
"dropout_10/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0"^dropout_9/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *N
fIRG
E__inference_dropout_10_layer_call_and_return_conditional_losses_391842$
"dropout_10/StatefulPartitionedCall?
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall+dropout_10/StatefulPartitionedCall:output:0conv2d_26_39225conv2d_26_39227*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_conv2d_26_layer_call_and_return_conditional_losses_392142#
!conv2d_26/StatefulPartitionedCall?
.batch_normalization_25/StatefulPartitionedCallStatefulPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0batch_normalization_25_39294batch_normalization_25_39296batch_normalization_25_39298batch_normalization_25_39300*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_25_layer_call_and_return_conditional_losses_3924920
.batch_normalization_25/StatefulPartitionedCall?
activation_25/PartitionedCallPartitionedCall7batch_normalization_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *Q
fLRJ
H__inference_activation_25_layer_call_and_return_conditional_losses_393082
activation_25/PartitionedCall?
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall&activation_25/PartitionedCall:output:0conv2d_transpose_5_39316conv2d_transpose_5_39318*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *V
fQRO
M__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_383182,
*conv2d_transpose_5/StatefulPartitionedCall?
concatenate_5/PartitionedCallPartitionedCall3conv2d_transpose_5/StatefulPartitionedCall:output:0&activation_15/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *Q
fLRJ
H__inference_concatenate_5_layer_call_and_return_conditional_losses_393282
concatenate_5/PartitionedCall?
"dropout_11/StatefulPartitionedCallStatefulPartitionedCall&concatenate_5/PartitionedCall:output:0#^dropout_10/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *N
fIRG
E__inference_dropout_11_layer_call_and_return_conditional_losses_393492$
"dropout_11/StatefulPartitionedCall?
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCall+dropout_11/StatefulPartitionedCall:output:0conv2d_28_39390conv2d_28_39392*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_conv2d_28_layer_call_and_return_conditional_losses_393792#
!conv2d_28/StatefulPartitionedCall?
.batch_normalization_27/StatefulPartitionedCallStatefulPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0batch_normalization_27_39459batch_normalization_27_39461batch_normalization_27_39463batch_normalization_27_39465*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_27_layer_call_and_return_conditional_losses_3941420
.batch_normalization_27/StatefulPartitionedCall?
activation_27/PartitionedCallPartitionedCall7batch_normalization_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *Q
fLRJ
H__inference_activation_27_layer_call_and_return_conditional_losses_394732
activation_27/PartitionedCall?
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall&activation_27/PartitionedCall:output:0conv2d_29_39505conv2d_29_39507*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_conv2d_29_layer_call_and_return_conditional_losses_394942#
!conv2d_29/StatefulPartitionedCall?
IdentityIdentity*conv2d_29/StatefulPartitionedCall:output:0/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall/^batch_normalization_19/StatefulPartitionedCall/^batch_normalization_21/StatefulPartitionedCall/^batch_normalization_23/StatefulPartitionedCall/^batch_normalization_25/StatefulPartitionedCall/^batch_normalization_27/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall"^conv2d_22/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall"^conv2d_28/StatefulPartitionedCall"^conv2d_29/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall+^conv2d_transpose_4/StatefulPartitionedCall+^conv2d_transpose_5/StatefulPartitionedCall#^dropout_10/StatefulPartitionedCall#^dropout_11/StatefulPartitionedCall"^dropout_6/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall"^dropout_9/StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2`
.batch_normalization_23/StatefulPartitionedCall.batch_normalization_23/StatefulPartitionedCall2`
.batch_normalization_25/StatefulPartitionedCall.batch_normalization_25/StatefulPartitionedCall2`
.batch_normalization_27/StatefulPartitionedCall.batch_normalization_27/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall2H
"dropout_10/StatefulPartitionedCall"dropout_10/StatefulPartitionedCall2H
"dropout_11/StatefulPartitionedCall"dropout_11/StatefulPartitionedCall2F
!dropout_6/StatefulPartitionedCall!dropout_6/StatefulPartitionedCall2F
!dropout_7/StatefulPartitionedCall!dropout_7/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall2F
!dropout_9/StatefulPartitionedCall!dropout_9/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_2
?
?
6__inference_batch_normalization_17_layer_call_fn_41370

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_386472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_21_layer_call_fn_41742

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????D\?*&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_21_layer_call_and_return_conditional_losses_389372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????D\?2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????D\?::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????D\?
 
_user_specified_nameinputs
?
d
H__inference_activation_17_layer_call_and_return_conditional_losses_41375

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:???????????@2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????@:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
d
H__inference_activation_15_layer_call_and_return_conditional_losses_41189

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:??????????? 2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
d
H__inference_activation_25_layer_call_and_return_conditional_losses_42145

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:???????????@2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????@:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_17_layer_call_fn_41293

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_377082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
D__inference_conv2d_24_layer_call_and_return_conditional_losses_41804

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*(
_output_shapes
:??2
Conv2D/Cast?
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:?2
BiasAdd/Cast}
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*2
_output_shapes 
:????????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_23_layer_call_and_return_conditional_losses_41915

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
K
/__inference_max_pooling2d_5_layer_call_fn_37878

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *S
fNRL
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_378722
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_38792

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
Y
-__inference_concatenate_4_layer_call_fn_41964
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *Q
fLRJ
H__inference_concatenate_4_layer_call_and_return_conditional_losses_391632
PartitionedCallw
IdentityIdentityPartitionedCall:output:0*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:+???????????????????????????@:???????????@:k g
A
_output_shapes/
-:+???????????????????????????@
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:???????????@
"
_user_specified_name
inputs/1
?
c
D__inference_dropout_9_layer_call_and_return_conditional_losses_39019

inputs
identity?a
dropout/ConstConst*
_output_shapes
: *
dtype0*
value
B j?x2
dropout/Const~
dropout/MulMulinputsdropout/Const:output:0*
T0*2
_output_shapes 
:????????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*2
_output_shapes 
:????????????*
dtype02&
$dropout/random_uniform/RandomUniforms
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value
B j?\2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*2
_output_shapes 
:????????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*2
_output_shapes 
:????????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*2
_output_shapes 
:????????????2
dropout/Mul_1p
IdentityIdentitydropout/Mul_1:z:0*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*1
_input_shapes 
:????????????:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
E
)__inference_dropout_8_layer_call_fn_41593

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????D\?* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_388592
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????D\?2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????D\?:X T
0
_output_shapes
:?????????D\?
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_23_layer_call_and_return_conditional_losses_41897

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_23_layer_call_fn_41928

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_23_layer_call_and_return_conditional_losses_390842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_17_layer_call_fn_41306

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_377392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_23_layer_call_and_return_conditional_losses_39084

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
d
H__inference_activation_19_layer_call_and_return_conditional_losses_41561

inputs
identityY
ReluReluinputs*
T0*2
_output_shapes 
:????????????2
Reluq
IdentityIdentityRelu:activations:0*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*1
_input_shapes 
:????????????:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_23_layer_call_and_return_conditional_losses_41851

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_38647

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_27_layer_call_and_return_conditional_losses_39414

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:??????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_23_layer_call_fn_41877

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_23_layer_call_and_return_conditional_losses_381212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
E
)__inference_dropout_9_layer_call_fn_41792

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_dropout_9_layer_call_and_return_conditional_losses_390242
PartitionedCallw
IdentityIdentityPartitionedCall:output:0*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*1
_input_shapes 
:????????????:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_21_layer_call_and_return_conditional_losses_38919

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????D\?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:?????????D\?2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????D\?::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????D\?
 
_user_specified_nameinputs
?
d
H__inference_activation_27_layer_call_and_return_conditional_losses_42344

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:??????????? 2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_27_layer_call_and_return_conditional_losses_42313

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
c
E__inference_dropout_11_layer_call_and_return_conditional_losses_42180

inputs

identity_1d
IdentityIdentityinputs*
T0*1
_output_shapes
:???????????@2

Identitys

Identity_1IdentityIdentity:output:0*
T0*1
_output_shapes
:???????????@2

Identity_1"!

identity_1Identity_1:output:0*0
_input_shapes
:???????????@:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_23_layer_call_and_return_conditional_losses_38090

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_21_layer_call_and_return_conditional_losses_41698

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????D\?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:?????????D\?2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????D\?::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????D\?
 
_user_specified_nameinputs
?
?
'__inference_model_1_layer_call_fn_39901
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*F
_read_only_resource_inputs(
&$	
!"#$%&)*+,-.12*8
config_proto(&

CPU

GPU2*0J

  ?E8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_397982
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_2
?
d
H__inference_activation_21_layer_call_and_return_conditional_losses_41747

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????D\?2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????D\?2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????D\?:X T
0
_output_shapes
:?????????D\?
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_23_layer_call_fn_41864

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_23_layer_call_and_return_conditional_losses_380902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
D__inference_conv2d_20_layer_call_and_return_conditional_losses_41419

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*'
_output_shapes
:@?2
Conv2D/Cast?
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:?2
BiasAdd/Cast}
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*2
_output_shapes 
:????????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
d
H__inference_activation_23_layer_call_and_return_conditional_losses_41946

inputs
identityY
ReluReluinputs*
T0*2
_output_shapes 
:????????????2
Reluq
IdentityIdentityRelu:activations:0*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*1
_input_shapes 
:????????????:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
b
D__inference_dropout_9_layer_call_and_return_conditional_losses_41782

inputs

identity_1e
IdentityIdentityinputs*
T0*2
_output_shapes 
:????????????2

Identityt

Identity_1IdentityIdentity:output:0*
T0*2
_output_shapes 
:????????????2

Identity_1"!

identity_1Identity_1:output:0*1
_input_shapes 
:????????????:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_25_layer_call_fn_42063

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_25_layer_call_and_return_conditional_losses_382402
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_25_layer_call_and_return_conditional_losses_39267

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_21_layer_call_fn_41729

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????D\?*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_21_layer_call_and_return_conditional_losses_389192
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????D\?2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????D\?::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????D\?
 
_user_specified_nameinputs
??
?
B__inference_model_1_layer_call_and_return_conditional_losses_40045

inputs
conv2d_16_39907
conv2d_16_39909 
batch_normalization_15_39912 
batch_normalization_15_39914 
batch_normalization_15_39916 
batch_normalization_15_39918
conv2d_18_39924
conv2d_18_39926 
batch_normalization_17_39929 
batch_normalization_17_39931 
batch_normalization_17_39933 
batch_normalization_17_39935
conv2d_20_39941
conv2d_20_39943 
batch_normalization_19_39946 
batch_normalization_19_39948 
batch_normalization_19_39950 
batch_normalization_19_39952
conv2d_22_39958
conv2d_22_39960 
batch_normalization_21_39963 
batch_normalization_21_39965 
batch_normalization_21_39967 
batch_normalization_21_39969
conv2d_transpose_3_39973
conv2d_transpose_3_39975
conv2d_24_39980
conv2d_24_39982 
batch_normalization_23_39985 
batch_normalization_23_39987 
batch_normalization_23_39989 
batch_normalization_23_39991
conv2d_transpose_4_39995
conv2d_transpose_4_39997
conv2d_26_40002
conv2d_26_40004 
batch_normalization_25_40007 
batch_normalization_25_40009 
batch_normalization_25_40011 
batch_normalization_25_40013
conv2d_transpose_5_40017
conv2d_transpose_5_40019
conv2d_28_40024
conv2d_28_40026 
batch_normalization_27_40029 
batch_normalization_27_40031 
batch_normalization_27_40033 
batch_normalization_27_40035
conv2d_29_40039
conv2d_29_40041
identity??.batch_normalization_15/StatefulPartitionedCall?.batch_normalization_17/StatefulPartitionedCall?.batch_normalization_19/StatefulPartitionedCall?.batch_normalization_21/StatefulPartitionedCall?.batch_normalization_23/StatefulPartitionedCall?.batch_normalization_25/StatefulPartitionedCall?.batch_normalization_27/StatefulPartitionedCall?!conv2d_16/StatefulPartitionedCall?!conv2d_18/StatefulPartitionedCall?!conv2d_20/StatefulPartitionedCall?!conv2d_22/StatefulPartitionedCall?!conv2d_24/StatefulPartitionedCall?!conv2d_26/StatefulPartitionedCall?!conv2d_28/StatefulPartitionedCall?!conv2d_29/StatefulPartitionedCall?*conv2d_transpose_3/StatefulPartitionedCall?*conv2d_transpose_4/StatefulPartitionedCall?*conv2d_transpose_5/StatefulPartitionedCallg
CastCastinputs*

DstT0*

SrcT0*1
_output_shapes
:???????????2
Cast?
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCallCast:y:0conv2d_16_39907conv2d_16_39909*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_conv2d_16_layer_call_and_return_conditional_losses_384492#
!conv2d_16/StatefulPartitionedCall?
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0batch_normalization_15_39912batch_normalization_15_39914batch_normalization_15_39916batch_normalization_15_39918*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_3850220
.batch_normalization_15/StatefulPartitionedCall?
activation_15/PartitionedCallPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *Q
fLRJ
H__inference_activation_15_layer_call_and_return_conditional_losses_385432
activation_15/PartitionedCall?
max_pooling2d_3/PartitionedCallPartitionedCall&activation_15/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_376402!
max_pooling2d_3/PartitionedCall?
dropout_6/PartitionedCallPartitionedCall(max_pooling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_385692
dropout_6/PartitionedCall?
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCall"dropout_6/PartitionedCall:output:0conv2d_18_39924conv2d_18_39926*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_conv2d_18_layer_call_and_return_conditional_losses_385942#
!conv2d_18/StatefulPartitionedCall?
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0batch_normalization_17_39929batch_normalization_17_39931batch_normalization_17_39933batch_normalization_17_39935*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_3864720
.batch_normalization_17/StatefulPartitionedCall?
activation_17/PartitionedCallPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *Q
fLRJ
H__inference_activation_17_layer_call_and_return_conditional_losses_386882
activation_17/PartitionedCall?
max_pooling2d_4/PartitionedCallPartitionedCall&activation_17/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_377562!
max_pooling2d_4/PartitionedCall?
dropout_7/PartitionedCallPartitionedCall(max_pooling2d_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_387142
dropout_7/PartitionedCall?
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall"dropout_7/PartitionedCall:output:0conv2d_20_39941conv2d_20_39943*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_conv2d_20_layer_call_and_return_conditional_losses_387392#
!conv2d_20/StatefulPartitionedCall?
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0batch_normalization_19_39946batch_normalization_19_39948batch_normalization_19_39950batch_normalization_19_39952*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_3879220
.batch_normalization_19/StatefulPartitionedCall?
activation_19/PartitionedCallPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *Q
fLRJ
H__inference_activation_19_layer_call_and_return_conditional_losses_388332
activation_19/PartitionedCall?
max_pooling2d_5/PartitionedCallPartitionedCall&activation_19/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????D\?* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *S
fNRL
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_378722!
max_pooling2d_5/PartitionedCall?
dropout_8/PartitionedCallPartitionedCall(max_pooling2d_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????D\?* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_388592
dropout_8/PartitionedCall?
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall"dropout_8/PartitionedCall:output:0conv2d_22_39958conv2d_22_39960*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????D\?*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_conv2d_22_layer_call_and_return_conditional_losses_388842#
!conv2d_22/StatefulPartitionedCall?
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0batch_normalization_21_39963batch_normalization_21_39965batch_normalization_21_39967batch_normalization_21_39969*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????D\?*&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_21_layer_call_and_return_conditional_losses_3893720
.batch_normalization_21/StatefulPartitionedCall?
activation_21/PartitionedCallPartitionedCall7batch_normalization_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????D\?* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *Q
fLRJ
H__inference_activation_21_layer_call_and_return_conditional_losses_389782
activation_21/PartitionedCall?
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall&activation_21/PartitionedCall:output:0conv2d_transpose_3_39973conv2d_transpose_3_39975*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *V
fQRO
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_380182,
*conv2d_transpose_3/StatefulPartitionedCall?
concatenate_3/PartitionedCallPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0&activation_19/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *Q
fLRJ
H__inference_concatenate_3_layer_call_and_return_conditional_losses_389982
concatenate_3/PartitionedCall?
dropout_9/PartitionedCallPartitionedCall&concatenate_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_dropout_9_layer_call_and_return_conditional_losses_390242
dropout_9/PartitionedCall?
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall"dropout_9/PartitionedCall:output:0conv2d_24_39980conv2d_24_39982*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_conv2d_24_layer_call_and_return_conditional_losses_390492#
!conv2d_24/StatefulPartitionedCall?
.batch_normalization_23/StatefulPartitionedCallStatefulPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0batch_normalization_23_39985batch_normalization_23_39987batch_normalization_23_39989batch_normalization_23_39991*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_23_layer_call_and_return_conditional_losses_3910220
.batch_normalization_23/StatefulPartitionedCall?
activation_23/PartitionedCallPartitionedCall7batch_normalization_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *Q
fLRJ
H__inference_activation_23_layer_call_and_return_conditional_losses_391432
activation_23/PartitionedCall?
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall&activation_23/PartitionedCall:output:0conv2d_transpose_4_39995conv2d_transpose_4_39997*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *V
fQRO
M__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_381682,
*conv2d_transpose_4/StatefulPartitionedCall?
concatenate_4/PartitionedCallPartitionedCall3conv2d_transpose_4/StatefulPartitionedCall:output:0&activation_17/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *Q
fLRJ
H__inference_concatenate_4_layer_call_and_return_conditional_losses_391632
concatenate_4/PartitionedCall?
dropout_10/PartitionedCallPartitionedCall&concatenate_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *N
fIRG
E__inference_dropout_10_layer_call_and_return_conditional_losses_391892
dropout_10/PartitionedCall?
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall#dropout_10/PartitionedCall:output:0conv2d_26_40002conv2d_26_40004*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_conv2d_26_layer_call_and_return_conditional_losses_392142#
!conv2d_26/StatefulPartitionedCall?
.batch_normalization_25/StatefulPartitionedCallStatefulPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0batch_normalization_25_40007batch_normalization_25_40009batch_normalization_25_40011batch_normalization_25_40013*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_25_layer_call_and_return_conditional_losses_3926720
.batch_normalization_25/StatefulPartitionedCall?
activation_25/PartitionedCallPartitionedCall7batch_normalization_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *Q
fLRJ
H__inference_activation_25_layer_call_and_return_conditional_losses_393082
activation_25/PartitionedCall?
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall&activation_25/PartitionedCall:output:0conv2d_transpose_5_40017conv2d_transpose_5_40019*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *V
fQRO
M__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_383182,
*conv2d_transpose_5/StatefulPartitionedCall?
concatenate_5/PartitionedCallPartitionedCall3conv2d_transpose_5/StatefulPartitionedCall:output:0&activation_15/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *Q
fLRJ
H__inference_concatenate_5_layer_call_and_return_conditional_losses_393282
concatenate_5/PartitionedCall?
dropout_11/PartitionedCallPartitionedCall&concatenate_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *N
fIRG
E__inference_dropout_11_layer_call_and_return_conditional_losses_393542
dropout_11/PartitionedCall?
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCall#dropout_11/PartitionedCall:output:0conv2d_28_40024conv2d_28_40026*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_conv2d_28_layer_call_and_return_conditional_losses_393792#
!conv2d_28/StatefulPartitionedCall?
.batch_normalization_27/StatefulPartitionedCallStatefulPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0batch_normalization_27_40029batch_normalization_27_40031batch_normalization_27_40033batch_normalization_27_40035*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_27_layer_call_and_return_conditional_losses_3943220
.batch_normalization_27/StatefulPartitionedCall?
activation_27/PartitionedCallPartitionedCall7batch_normalization_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *Q
fLRJ
H__inference_activation_27_layer_call_and_return_conditional_losses_394732
activation_27/PartitionedCall?
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall&activation_27/PartitionedCall:output:0conv2d_29_40039conv2d_29_40041*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_conv2d_29_layer_call_and_return_conditional_losses_394942#
!conv2d_29/StatefulPartitionedCall?
IdentityIdentity*conv2d_29/StatefulPartitionedCall:output:0/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall/^batch_normalization_19/StatefulPartitionedCall/^batch_normalization_21/StatefulPartitionedCall/^batch_normalization_23/StatefulPartitionedCall/^batch_normalization_25/StatefulPartitionedCall/^batch_normalization_27/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall"^conv2d_22/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall"^conv2d_28/StatefulPartitionedCall"^conv2d_29/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall+^conv2d_transpose_4/StatefulPartitionedCall+^conv2d_transpose_5/StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2`
.batch_normalization_23/StatefulPartitionedCall.batch_normalization_23/StatefulPartitionedCall2`
.batch_normalization_25/StatefulPartitionedCall.batch_normalization_25/StatefulPartitionedCall2`
.batch_normalization_27/StatefulPartitionedCall.batch_normalization_27/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_21_layer_call_and_return_conditional_losses_41716

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????D\?:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:?????????D\?2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????D\?::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????D\?
 
_user_specified_nameinputs
?
F
*__inference_dropout_10_layer_call_fn_41991

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *N
fIRG
E__inference_dropout_10_layer_call_and_return_conditional_losses_391892
PartitionedCallw
IdentityIdentityPartitionedCall:output:0*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*1
_input_shapes 
:????????????:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_41326

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
b
D__inference_dropout_7_layer_call_and_return_conditional_losses_41397

inputs

identity_1d
IdentityIdentityinputs*
T0*1
_output_shapes
:???????????@2

Identitys

Identity_1IdentityIdentity:output:0*
T0*1
_output_shapes
:???????????@2

Identity_1"!

identity_1Identity_1:output:0*0
_input_shapes
:???????????@:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_41094

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
d
E__inference_dropout_11_layer_call_and_return_conditional_losses_42175

inputs
identity?a
dropout/ConstConst*
_output_shapes
: *
dtype0*
value
B j?x2
dropout/Const}
dropout/MulMulinputsdropout/Const:output:0*
T0*1
_output_shapes
:???????????@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*1
_output_shapes
:???????????@*
dtype02&
$dropout/random_uniform/RandomUniforms
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value
B j?\2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:???????????@2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:???????????@2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*1
_output_shapes
:???????????@2
dropout/Mul_1o
IdentityIdentitydropout/Mul_1:z:0*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????@:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_41262

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
??
?B
__inference__traced_save_42793
file_prefix/
+savev2_conv2d_16_kernel_read_readvariableop-
)savev2_conv2d_16_bias_read_readvariableop;
7savev2_batch_normalization_15_gamma_read_readvariableop:
6savev2_batch_normalization_15_beta_read_readvariableopA
=savev2_batch_normalization_15_moving_mean_read_readvariableopE
Asavev2_batch_normalization_15_moving_variance_read_readvariableop/
+savev2_conv2d_18_kernel_read_readvariableop-
)savev2_conv2d_18_bias_read_readvariableop;
7savev2_batch_normalization_17_gamma_read_readvariableop:
6savev2_batch_normalization_17_beta_read_readvariableopA
=savev2_batch_normalization_17_moving_mean_read_readvariableopE
Asavev2_batch_normalization_17_moving_variance_read_readvariableop/
+savev2_conv2d_20_kernel_read_readvariableop-
)savev2_conv2d_20_bias_read_readvariableop;
7savev2_batch_normalization_19_gamma_read_readvariableop:
6savev2_batch_normalization_19_beta_read_readvariableopA
=savev2_batch_normalization_19_moving_mean_read_readvariableopE
Asavev2_batch_normalization_19_moving_variance_read_readvariableop/
+savev2_conv2d_22_kernel_read_readvariableop-
)savev2_conv2d_22_bias_read_readvariableop;
7savev2_batch_normalization_21_gamma_read_readvariableop:
6savev2_batch_normalization_21_beta_read_readvariableopA
=savev2_batch_normalization_21_moving_mean_read_readvariableopE
Asavev2_batch_normalization_21_moving_variance_read_readvariableop8
4savev2_conv2d_transpose_3_kernel_read_readvariableop6
2savev2_conv2d_transpose_3_bias_read_readvariableop/
+savev2_conv2d_24_kernel_read_readvariableop-
)savev2_conv2d_24_bias_read_readvariableop;
7savev2_batch_normalization_23_gamma_read_readvariableop:
6savev2_batch_normalization_23_beta_read_readvariableopA
=savev2_batch_normalization_23_moving_mean_read_readvariableopE
Asavev2_batch_normalization_23_moving_variance_read_readvariableop8
4savev2_conv2d_transpose_4_kernel_read_readvariableop6
2savev2_conv2d_transpose_4_bias_read_readvariableop/
+savev2_conv2d_26_kernel_read_readvariableop-
)savev2_conv2d_26_bias_read_readvariableop;
7savev2_batch_normalization_25_gamma_read_readvariableop:
6savev2_batch_normalization_25_beta_read_readvariableopA
=savev2_batch_normalization_25_moving_mean_read_readvariableopE
Asavev2_batch_normalization_25_moving_variance_read_readvariableop8
4savev2_conv2d_transpose_5_kernel_read_readvariableop6
2savev2_conv2d_transpose_5_bias_read_readvariableop/
+savev2_conv2d_28_kernel_read_readvariableop-
)savev2_conv2d_28_bias_read_readvariableop;
7savev2_batch_normalization_27_gamma_read_readvariableop:
6savev2_batch_normalization_27_beta_read_readvariableopA
=savev2_batch_normalization_27_moving_mean_read_readvariableopE
Asavev2_batch_normalization_27_moving_variance_read_readvariableop/
+savev2_conv2d_29_kernel_read_readvariableop-
)savev2_conv2d_29_bias_read_readvariableop/
+savev2_cond_1_adam_iter_read_readvariableop	1
-savev2_cond_1_adam_beta_1_read_readvariableop1
-savev2_cond_1_adam_beta_2_read_readvariableop0
,savev2_cond_1_adam_decay_read_readvariableop8
4savev2_cond_1_adam_learning_rate_read_readvariableop1
-savev2_current_loss_scale_read_readvariableop)
%savev2_good_steps_read_readvariableop	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop=
9savev2_cond_1_adam_conv2d_16_kernel_m_read_readvariableop;
7savev2_cond_1_adam_conv2d_16_bias_m_read_readvariableopI
Esavev2_cond_1_adam_batch_normalization_15_gamma_m_read_readvariableopH
Dsavev2_cond_1_adam_batch_normalization_15_beta_m_read_readvariableop=
9savev2_cond_1_adam_conv2d_18_kernel_m_read_readvariableop;
7savev2_cond_1_adam_conv2d_18_bias_m_read_readvariableopI
Esavev2_cond_1_adam_batch_normalization_17_gamma_m_read_readvariableopH
Dsavev2_cond_1_adam_batch_normalization_17_beta_m_read_readvariableop=
9savev2_cond_1_adam_conv2d_20_kernel_m_read_readvariableop;
7savev2_cond_1_adam_conv2d_20_bias_m_read_readvariableopI
Esavev2_cond_1_adam_batch_normalization_19_gamma_m_read_readvariableopH
Dsavev2_cond_1_adam_batch_normalization_19_beta_m_read_readvariableop=
9savev2_cond_1_adam_conv2d_22_kernel_m_read_readvariableop;
7savev2_cond_1_adam_conv2d_22_bias_m_read_readvariableopI
Esavev2_cond_1_adam_batch_normalization_21_gamma_m_read_readvariableopH
Dsavev2_cond_1_adam_batch_normalization_21_beta_m_read_readvariableopF
Bsavev2_cond_1_adam_conv2d_transpose_3_kernel_m_read_readvariableopD
@savev2_cond_1_adam_conv2d_transpose_3_bias_m_read_readvariableop=
9savev2_cond_1_adam_conv2d_24_kernel_m_read_readvariableop;
7savev2_cond_1_adam_conv2d_24_bias_m_read_readvariableopI
Esavev2_cond_1_adam_batch_normalization_23_gamma_m_read_readvariableopH
Dsavev2_cond_1_adam_batch_normalization_23_beta_m_read_readvariableopF
Bsavev2_cond_1_adam_conv2d_transpose_4_kernel_m_read_readvariableopD
@savev2_cond_1_adam_conv2d_transpose_4_bias_m_read_readvariableop=
9savev2_cond_1_adam_conv2d_26_kernel_m_read_readvariableop;
7savev2_cond_1_adam_conv2d_26_bias_m_read_readvariableopI
Esavev2_cond_1_adam_batch_normalization_25_gamma_m_read_readvariableopH
Dsavev2_cond_1_adam_batch_normalization_25_beta_m_read_readvariableopF
Bsavev2_cond_1_adam_conv2d_transpose_5_kernel_m_read_readvariableopD
@savev2_cond_1_adam_conv2d_transpose_5_bias_m_read_readvariableop=
9savev2_cond_1_adam_conv2d_28_kernel_m_read_readvariableop;
7savev2_cond_1_adam_conv2d_28_bias_m_read_readvariableopI
Esavev2_cond_1_adam_batch_normalization_27_gamma_m_read_readvariableopH
Dsavev2_cond_1_adam_batch_normalization_27_beta_m_read_readvariableop=
9savev2_cond_1_adam_conv2d_29_kernel_m_read_readvariableop;
7savev2_cond_1_adam_conv2d_29_bias_m_read_readvariableop=
9savev2_cond_1_adam_conv2d_16_kernel_v_read_readvariableop;
7savev2_cond_1_adam_conv2d_16_bias_v_read_readvariableopI
Esavev2_cond_1_adam_batch_normalization_15_gamma_v_read_readvariableopH
Dsavev2_cond_1_adam_batch_normalization_15_beta_v_read_readvariableop=
9savev2_cond_1_adam_conv2d_18_kernel_v_read_readvariableop;
7savev2_cond_1_adam_conv2d_18_bias_v_read_readvariableopI
Esavev2_cond_1_adam_batch_normalization_17_gamma_v_read_readvariableopH
Dsavev2_cond_1_adam_batch_normalization_17_beta_v_read_readvariableop=
9savev2_cond_1_adam_conv2d_20_kernel_v_read_readvariableop;
7savev2_cond_1_adam_conv2d_20_bias_v_read_readvariableopI
Esavev2_cond_1_adam_batch_normalization_19_gamma_v_read_readvariableopH
Dsavev2_cond_1_adam_batch_normalization_19_beta_v_read_readvariableop=
9savev2_cond_1_adam_conv2d_22_kernel_v_read_readvariableop;
7savev2_cond_1_adam_conv2d_22_bias_v_read_readvariableopI
Esavev2_cond_1_adam_batch_normalization_21_gamma_v_read_readvariableopH
Dsavev2_cond_1_adam_batch_normalization_21_beta_v_read_readvariableopF
Bsavev2_cond_1_adam_conv2d_transpose_3_kernel_v_read_readvariableopD
@savev2_cond_1_adam_conv2d_transpose_3_bias_v_read_readvariableop=
9savev2_cond_1_adam_conv2d_24_kernel_v_read_readvariableop;
7savev2_cond_1_adam_conv2d_24_bias_v_read_readvariableopI
Esavev2_cond_1_adam_batch_normalization_23_gamma_v_read_readvariableopH
Dsavev2_cond_1_adam_batch_normalization_23_beta_v_read_readvariableopF
Bsavev2_cond_1_adam_conv2d_transpose_4_kernel_v_read_readvariableopD
@savev2_cond_1_adam_conv2d_transpose_4_bias_v_read_readvariableop=
9savev2_cond_1_adam_conv2d_26_kernel_v_read_readvariableop;
7savev2_cond_1_adam_conv2d_26_bias_v_read_readvariableopI
Esavev2_cond_1_adam_batch_normalization_25_gamma_v_read_readvariableopH
Dsavev2_cond_1_adam_batch_normalization_25_beta_v_read_readvariableopF
Bsavev2_cond_1_adam_conv2d_transpose_5_kernel_v_read_readvariableopD
@savev2_cond_1_adam_conv2d_transpose_5_bias_v_read_readvariableop=
9savev2_cond_1_adam_conv2d_28_kernel_v_read_readvariableop;
7savev2_cond_1_adam_conv2d_28_bias_v_read_readvariableopI
Esavev2_cond_1_adam_batch_normalization_27_gamma_v_read_readvariableopH
Dsavev2_cond_1_adam_batch_normalization_27_beta_v_read_readvariableop=
9savev2_cond_1_adam_conv2d_29_kernel_v_read_readvariableop;
7savev2_cond_1_adam_conv2d_29_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?K
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?J
value?JB?J?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-16/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-16/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBBoptimizer/loss_scale/current_loss_scale/.ATTRIBUTES/VARIABLE_VALUEB:optimizer/loss_scale/good_steps/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices??
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_16_kernel_read_readvariableop)savev2_conv2d_16_bias_read_readvariableop7savev2_batch_normalization_15_gamma_read_readvariableop6savev2_batch_normalization_15_beta_read_readvariableop=savev2_batch_normalization_15_moving_mean_read_readvariableopAsavev2_batch_normalization_15_moving_variance_read_readvariableop+savev2_conv2d_18_kernel_read_readvariableop)savev2_conv2d_18_bias_read_readvariableop7savev2_batch_normalization_17_gamma_read_readvariableop6savev2_batch_normalization_17_beta_read_readvariableop=savev2_batch_normalization_17_moving_mean_read_readvariableopAsavev2_batch_normalization_17_moving_variance_read_readvariableop+savev2_conv2d_20_kernel_read_readvariableop)savev2_conv2d_20_bias_read_readvariableop7savev2_batch_normalization_19_gamma_read_readvariableop6savev2_batch_normalization_19_beta_read_readvariableop=savev2_batch_normalization_19_moving_mean_read_readvariableopAsavev2_batch_normalization_19_moving_variance_read_readvariableop+savev2_conv2d_22_kernel_read_readvariableop)savev2_conv2d_22_bias_read_readvariableop7savev2_batch_normalization_21_gamma_read_readvariableop6savev2_batch_normalization_21_beta_read_readvariableop=savev2_batch_normalization_21_moving_mean_read_readvariableopAsavev2_batch_normalization_21_moving_variance_read_readvariableop4savev2_conv2d_transpose_3_kernel_read_readvariableop2savev2_conv2d_transpose_3_bias_read_readvariableop+savev2_conv2d_24_kernel_read_readvariableop)savev2_conv2d_24_bias_read_readvariableop7savev2_batch_normalization_23_gamma_read_readvariableop6savev2_batch_normalization_23_beta_read_readvariableop=savev2_batch_normalization_23_moving_mean_read_readvariableopAsavev2_batch_normalization_23_moving_variance_read_readvariableop4savev2_conv2d_transpose_4_kernel_read_readvariableop2savev2_conv2d_transpose_4_bias_read_readvariableop+savev2_conv2d_26_kernel_read_readvariableop)savev2_conv2d_26_bias_read_readvariableop7savev2_batch_normalization_25_gamma_read_readvariableop6savev2_batch_normalization_25_beta_read_readvariableop=savev2_batch_normalization_25_moving_mean_read_readvariableopAsavev2_batch_normalization_25_moving_variance_read_readvariableop4savev2_conv2d_transpose_5_kernel_read_readvariableop2savev2_conv2d_transpose_5_bias_read_readvariableop+savev2_conv2d_28_kernel_read_readvariableop)savev2_conv2d_28_bias_read_readvariableop7savev2_batch_normalization_27_gamma_read_readvariableop6savev2_batch_normalization_27_beta_read_readvariableop=savev2_batch_normalization_27_moving_mean_read_readvariableopAsavev2_batch_normalization_27_moving_variance_read_readvariableop+savev2_conv2d_29_kernel_read_readvariableop)savev2_conv2d_29_bias_read_readvariableop+savev2_cond_1_adam_iter_read_readvariableop-savev2_cond_1_adam_beta_1_read_readvariableop-savev2_cond_1_adam_beta_2_read_readvariableop,savev2_cond_1_adam_decay_read_readvariableop4savev2_cond_1_adam_learning_rate_read_readvariableop-savev2_current_loss_scale_read_readvariableop%savev2_good_steps_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop9savev2_cond_1_adam_conv2d_16_kernel_m_read_readvariableop7savev2_cond_1_adam_conv2d_16_bias_m_read_readvariableopEsavev2_cond_1_adam_batch_normalization_15_gamma_m_read_readvariableopDsavev2_cond_1_adam_batch_normalization_15_beta_m_read_readvariableop9savev2_cond_1_adam_conv2d_18_kernel_m_read_readvariableop7savev2_cond_1_adam_conv2d_18_bias_m_read_readvariableopEsavev2_cond_1_adam_batch_normalization_17_gamma_m_read_readvariableopDsavev2_cond_1_adam_batch_normalization_17_beta_m_read_readvariableop9savev2_cond_1_adam_conv2d_20_kernel_m_read_readvariableop7savev2_cond_1_adam_conv2d_20_bias_m_read_readvariableopEsavev2_cond_1_adam_batch_normalization_19_gamma_m_read_readvariableopDsavev2_cond_1_adam_batch_normalization_19_beta_m_read_readvariableop9savev2_cond_1_adam_conv2d_22_kernel_m_read_readvariableop7savev2_cond_1_adam_conv2d_22_bias_m_read_readvariableopEsavev2_cond_1_adam_batch_normalization_21_gamma_m_read_readvariableopDsavev2_cond_1_adam_batch_normalization_21_beta_m_read_readvariableopBsavev2_cond_1_adam_conv2d_transpose_3_kernel_m_read_readvariableop@savev2_cond_1_adam_conv2d_transpose_3_bias_m_read_readvariableop9savev2_cond_1_adam_conv2d_24_kernel_m_read_readvariableop7savev2_cond_1_adam_conv2d_24_bias_m_read_readvariableopEsavev2_cond_1_adam_batch_normalization_23_gamma_m_read_readvariableopDsavev2_cond_1_adam_batch_normalization_23_beta_m_read_readvariableopBsavev2_cond_1_adam_conv2d_transpose_4_kernel_m_read_readvariableop@savev2_cond_1_adam_conv2d_transpose_4_bias_m_read_readvariableop9savev2_cond_1_adam_conv2d_26_kernel_m_read_readvariableop7savev2_cond_1_adam_conv2d_26_bias_m_read_readvariableopEsavev2_cond_1_adam_batch_normalization_25_gamma_m_read_readvariableopDsavev2_cond_1_adam_batch_normalization_25_beta_m_read_readvariableopBsavev2_cond_1_adam_conv2d_transpose_5_kernel_m_read_readvariableop@savev2_cond_1_adam_conv2d_transpose_5_bias_m_read_readvariableop9savev2_cond_1_adam_conv2d_28_kernel_m_read_readvariableop7savev2_cond_1_adam_conv2d_28_bias_m_read_readvariableopEsavev2_cond_1_adam_batch_normalization_27_gamma_m_read_readvariableopDsavev2_cond_1_adam_batch_normalization_27_beta_m_read_readvariableop9savev2_cond_1_adam_conv2d_29_kernel_m_read_readvariableop7savev2_cond_1_adam_conv2d_29_bias_m_read_readvariableop9savev2_cond_1_adam_conv2d_16_kernel_v_read_readvariableop7savev2_cond_1_adam_conv2d_16_bias_v_read_readvariableopEsavev2_cond_1_adam_batch_normalization_15_gamma_v_read_readvariableopDsavev2_cond_1_adam_batch_normalization_15_beta_v_read_readvariableop9savev2_cond_1_adam_conv2d_18_kernel_v_read_readvariableop7savev2_cond_1_adam_conv2d_18_bias_v_read_readvariableopEsavev2_cond_1_adam_batch_normalization_17_gamma_v_read_readvariableopDsavev2_cond_1_adam_batch_normalization_17_beta_v_read_readvariableop9savev2_cond_1_adam_conv2d_20_kernel_v_read_readvariableop7savev2_cond_1_adam_conv2d_20_bias_v_read_readvariableopEsavev2_cond_1_adam_batch_normalization_19_gamma_v_read_readvariableopDsavev2_cond_1_adam_batch_normalization_19_beta_v_read_readvariableop9savev2_cond_1_adam_conv2d_22_kernel_v_read_readvariableop7savev2_cond_1_adam_conv2d_22_bias_v_read_readvariableopEsavev2_cond_1_adam_batch_normalization_21_gamma_v_read_readvariableopDsavev2_cond_1_adam_batch_normalization_21_beta_v_read_readvariableopBsavev2_cond_1_adam_conv2d_transpose_3_kernel_v_read_readvariableop@savev2_cond_1_adam_conv2d_transpose_3_bias_v_read_readvariableop9savev2_cond_1_adam_conv2d_24_kernel_v_read_readvariableop7savev2_cond_1_adam_conv2d_24_bias_v_read_readvariableopEsavev2_cond_1_adam_batch_normalization_23_gamma_v_read_readvariableopDsavev2_cond_1_adam_batch_normalization_23_beta_v_read_readvariableopBsavev2_cond_1_adam_conv2d_transpose_4_kernel_v_read_readvariableop@savev2_cond_1_adam_conv2d_transpose_4_bias_v_read_readvariableop9savev2_cond_1_adam_conv2d_26_kernel_v_read_readvariableop7savev2_cond_1_adam_conv2d_26_bias_v_read_readvariableopEsavev2_cond_1_adam_batch_normalization_25_gamma_v_read_readvariableopDsavev2_cond_1_adam_batch_normalization_25_beta_v_read_readvariableopBsavev2_cond_1_adam_conv2d_transpose_5_kernel_v_read_readvariableop@savev2_cond_1_adam_conv2d_transpose_5_bias_v_read_readvariableop9savev2_cond_1_adam_conv2d_28_kernel_v_read_readvariableop7savev2_cond_1_adam_conv2d_28_bias_v_read_readvariableopEsavev2_cond_1_adam_batch_normalization_27_gamma_v_read_readvariableopDsavev2_cond_1_adam_batch_normalization_27_beta_v_read_readvariableop9savev2_cond_1_adam_conv2d_29_kernel_v_read_readvariableop7savev2_cond_1_adam_conv2d_29_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *?
dtypes?
?2?		2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?	
_input_shapes?	
?	: : : : : : : : @:@:@:@:@:@:@?:?:?:?:?:?:??:?:?:?:?:?:??:?:??:?:?:?:?:?:@?:@:?@:@:@:@:@:@: @: :@ : : : : : : :: : : : : : : : : : : : : : : : @:@:@:@:@?:?:?:?:??:?:?:?:??:?:??:?:?:?:@?:@:?@:@:@:@: @: :@ : : : : :: : : : : @:@:@:@:@?:?:?:?:??:?:?:?:??:?:??:?:?:?:@?:@:?@:@:@:@: @: :@ : : : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@: 	

_output_shapes
:@: 


_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:@?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:! 

_output_shapes	
:?:-!)
'
_output_shapes
:@?: "

_output_shapes
:@:-#)
'
_output_shapes
:?@: $

_output_shapes
:@: %

_output_shapes
:@: &

_output_shapes
:@: '

_output_shapes
:@: (

_output_shapes
:@:,)(
&
_output_shapes
: @: *

_output_shapes
: :,+(
&
_output_shapes
:@ : ,

_output_shapes
: : -

_output_shapes
: : .

_output_shapes
: : /

_output_shapes
: : 0

_output_shapes
: :,1(
&
_output_shapes
: : 2

_output_shapes
::3

_output_shapes
: :4

_output_shapes
: :5

_output_shapes
: :6

_output_shapes
: :7

_output_shapes
: :8

_output_shapes
: :9

_output_shapes
: ::

_output_shapes
: :;

_output_shapes
: :<

_output_shapes
: :=

_output_shapes
: :,>(
&
_output_shapes
: : ?

_output_shapes
: : @

_output_shapes
: : A

_output_shapes
: :,B(
&
_output_shapes
: @: C

_output_shapes
:@: D

_output_shapes
:@: E

_output_shapes
:@:-F)
'
_output_shapes
:@?:!G

_output_shapes	
:?:!H

_output_shapes	
:?:!I

_output_shapes	
:?:.J*
(
_output_shapes
:??:!K

_output_shapes	
:?:!L

_output_shapes	
:?:!M

_output_shapes	
:?:.N*
(
_output_shapes
:??:!O

_output_shapes	
:?:.P*
(
_output_shapes
:??:!Q

_output_shapes	
:?:!R

_output_shapes	
:?:!S

_output_shapes	
:?:-T)
'
_output_shapes
:@?: U

_output_shapes
:@:-V)
'
_output_shapes
:?@: W

_output_shapes
:@: X

_output_shapes
:@: Y

_output_shapes
:@:,Z(
&
_output_shapes
: @: [

_output_shapes
: :,\(
&
_output_shapes
:@ : ]

_output_shapes
: : ^

_output_shapes
: : _

_output_shapes
: :,`(
&
_output_shapes
: : a

_output_shapes
::,b(
&
_output_shapes
: : c

_output_shapes
: : d

_output_shapes
: : e

_output_shapes
: :,f(
&
_output_shapes
: @: g

_output_shapes
:@: h

_output_shapes
:@: i

_output_shapes
:@:-j)
'
_output_shapes
:@?:!k

_output_shapes	
:?:!l

_output_shapes	
:?:!m

_output_shapes	
:?:.n*
(
_output_shapes
:??:!o

_output_shapes	
:?:!p

_output_shapes	
:?:!q

_output_shapes	
:?:.r*
(
_output_shapes
:??:!s

_output_shapes	
:?:.t*
(
_output_shapes
:??:!u

_output_shapes	
:?:!v

_output_shapes	
:?:!w

_output_shapes	
:?:-x)
'
_output_shapes
:@?: y

_output_shapes
:@:-z)
'
_output_shapes
:?@: {

_output_shapes
:@: |

_output_shapes
:@: }

_output_shapes
:@:,~(
&
_output_shapes
: @: 

_output_shapes
: :-?(
&
_output_shapes
:@ :!?

_output_shapes
: :!?

_output_shapes
: :!?

_output_shapes
: :-?(
&
_output_shapes
: :!?

_output_shapes
::?

_output_shapes
: 
?
?
Q__inference_batch_normalization_27_layer_call_and_return_conditional_losses_42249

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:??????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
b
)__inference_dropout_9_layer_call_fn_41787

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_dropout_9_layer_call_and_return_conditional_losses_390192
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*1
_input_shapes 
:????????????22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
E
)__inference_dropout_6_layer_call_fn_41221

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_385692
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
D__inference_conv2d_28_layer_call_and_return_conditional_losses_39379

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02
Conv2D/ReadVariableOp?
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:@ 2
Conv2D/Cast?
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpx
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: 2
BiasAdd/Cast|
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*1
_output_shapes
:??????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_19_layer_call_fn_41492

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_378552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
c
D__inference_dropout_6_layer_call_and_return_conditional_losses_41206

inputs
identity?a
dropout/ConstConst*
_output_shapes
: *
dtype0*
value
B j?x2
dropout/Const}
dropout/MulMulinputsdropout/Const:output:0*
T0*1
_output_shapes
:??????????? 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*1
_output_shapes
:??????????? *
dtype02&
$dropout/random_uniform/RandomUniforms
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value
B j?\2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:??????????? 2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:??????????? 2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*1
_output_shapes
:??????????? 2
dropout/Mul_1o
IdentityIdentitydropout/Mul_1:z:0*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
2__inference_conv2d_transpose_3_layer_call_fn_38028

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *V
fQRO
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_380182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_10_layer_call_and_return_conditional_losses_41981

inputs

identity_1e
IdentityIdentityinputs*
T0*2
_output_shapes 
:????????????2

Identityt

Identity_1IdentityIdentity:output:0*
T0*2
_output_shapes 
:????????????2

Identity_1"!

identity_1Identity_1:output:0*1
_input_shapes 
:????????????:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
d
H__inference_activation_23_layer_call_and_return_conditional_losses_39143

inputs
identityY
ReluReluinputs*
T0*2
_output_shapes 
:????????????2
Reluq
IdentityIdentityRelu:activations:0*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*1
_input_shapes 
:????????????:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
K
/__inference_max_pooling2d_4_layer_call_fn_37762

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_377562
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
~
)__inference_conv2d_29_layer_call_fn_42371

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_conv2d_29_layer_call_and_return_conditional_losses_394942
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:??????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_27_layer_call_and_return_conditional_losses_38421

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
I
-__inference_activation_27_layer_call_fn_42349

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *Q
fLRJ
H__inference_activation_27_layer_call_and_return_conditional_losses_394732
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
??
?,
B__inference_model_1_layer_call_and_return_conditional_losses_40572

inputs,
(conv2d_16_conv2d_readvariableop_resource-
)conv2d_16_biasadd_readvariableop_resource2
.batch_normalization_15_readvariableop_resource4
0batch_normalization_15_readvariableop_1_resourceC
?batch_normalization_15_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_18_conv2d_readvariableop_resource-
)conv2d_18_biasadd_readvariableop_resource2
.batch_normalization_17_readvariableop_resource4
0batch_normalization_17_readvariableop_1_resourceC
?batch_normalization_17_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_20_conv2d_readvariableop_resource-
)conv2d_20_biasadd_readvariableop_resource2
.batch_normalization_19_readvariableop_resource4
0batch_normalization_19_readvariableop_1_resourceC
?batch_normalization_19_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_22_conv2d_readvariableop_resource-
)conv2d_22_biasadd_readvariableop_resource2
.batch_normalization_21_readvariableop_resource4
0batch_normalization_21_readvariableop_1_resourceC
?batch_normalization_21_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource?
;conv2d_transpose_3_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_3_biasadd_readvariableop_resource,
(conv2d_24_conv2d_readvariableop_resource-
)conv2d_24_biasadd_readvariableop_resource2
.batch_normalization_23_readvariableop_resource4
0batch_normalization_23_readvariableop_1_resourceC
?batch_normalization_23_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource?
;conv2d_transpose_4_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_4_biasadd_readvariableop_resource,
(conv2d_26_conv2d_readvariableop_resource-
)conv2d_26_biasadd_readvariableop_resource2
.batch_normalization_25_readvariableop_resource4
0batch_normalization_25_readvariableop_1_resourceC
?batch_normalization_25_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_25_fusedbatchnormv3_readvariableop_1_resource?
;conv2d_transpose_5_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_5_biasadd_readvariableop_resource,
(conv2d_28_conv2d_readvariableop_resource-
)conv2d_28_biasadd_readvariableop_resource2
.batch_normalization_27_readvariableop_resource4
0batch_normalization_27_readvariableop_1_resourceC
?batch_normalization_27_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_29_conv2d_readvariableop_resource-
)conv2d_29_biasadd_readvariableop_resource
identity??%batch_normalization_15/AssignNewValue?'batch_normalization_15/AssignNewValue_1?6batch_normalization_15/FusedBatchNormV3/ReadVariableOp?8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_15/ReadVariableOp?'batch_normalization_15/ReadVariableOp_1?%batch_normalization_17/AssignNewValue?'batch_normalization_17/AssignNewValue_1?6batch_normalization_17/FusedBatchNormV3/ReadVariableOp?8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_17/ReadVariableOp?'batch_normalization_17/ReadVariableOp_1?%batch_normalization_19/AssignNewValue?'batch_normalization_19/AssignNewValue_1?6batch_normalization_19/FusedBatchNormV3/ReadVariableOp?8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_19/ReadVariableOp?'batch_normalization_19/ReadVariableOp_1?%batch_normalization_21/AssignNewValue?'batch_normalization_21/AssignNewValue_1?6batch_normalization_21/FusedBatchNormV3/ReadVariableOp?8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_21/ReadVariableOp?'batch_normalization_21/ReadVariableOp_1?%batch_normalization_23/AssignNewValue?'batch_normalization_23/AssignNewValue_1?6batch_normalization_23/FusedBatchNormV3/ReadVariableOp?8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_23/ReadVariableOp?'batch_normalization_23/ReadVariableOp_1?%batch_normalization_25/AssignNewValue?'batch_normalization_25/AssignNewValue_1?6batch_normalization_25/FusedBatchNormV3/ReadVariableOp?8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_25/ReadVariableOp?'batch_normalization_25/ReadVariableOp_1?%batch_normalization_27/AssignNewValue?'batch_normalization_27/AssignNewValue_1?6batch_normalization_27/FusedBatchNormV3/ReadVariableOp?8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_27/ReadVariableOp?'batch_normalization_27/ReadVariableOp_1? conv2d_16/BiasAdd/ReadVariableOp?conv2d_16/Conv2D/ReadVariableOp? conv2d_18/BiasAdd/ReadVariableOp?conv2d_18/Conv2D/ReadVariableOp? conv2d_20/BiasAdd/ReadVariableOp?conv2d_20/Conv2D/ReadVariableOp? conv2d_22/BiasAdd/ReadVariableOp?conv2d_22/Conv2D/ReadVariableOp? conv2d_24/BiasAdd/ReadVariableOp?conv2d_24/Conv2D/ReadVariableOp? conv2d_26/BiasAdd/ReadVariableOp?conv2d_26/Conv2D/ReadVariableOp? conv2d_28/BiasAdd/ReadVariableOp?conv2d_28/Conv2D/ReadVariableOp? conv2d_29/BiasAdd/ReadVariableOp?conv2d_29/Conv2D/ReadVariableOp?)conv2d_transpose_3/BiasAdd/ReadVariableOp?2conv2d_transpose_3/conv2d_transpose/ReadVariableOp?)conv2d_transpose_4/BiasAdd/ReadVariableOp?2conv2d_transpose_4/conv2d_transpose/ReadVariableOp?)conv2d_transpose_5/BiasAdd/ReadVariableOp?2conv2d_transpose_5/conv2d_transpose/ReadVariableOpg
CastCastinputs*

DstT0*

SrcT0*1
_output_shapes
:???????????2
Cast?
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_16/Conv2D/ReadVariableOp?
conv2d_16/Conv2D/CastCast'conv2d_16/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: 2
conv2d_16/Conv2D/Cast?
conv2d_16/Conv2DConv2DCast:y:0conv2d_16/Conv2D/Cast:y:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
conv2d_16/Conv2D?
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_16/BiasAdd/ReadVariableOp?
conv2d_16/BiasAdd/CastCast(conv2d_16/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: 2
conv2d_16/BiasAdd/Cast?
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0conv2d_16/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:??????????? 2
conv2d_16/BiasAdd?
%batch_normalization_15/ReadVariableOpReadVariableOp.batch_normalization_15_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_15/ReadVariableOp?
'batch_normalization_15/ReadVariableOp_1ReadVariableOp0batch_normalization_15_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_15/ReadVariableOp_1?
6batch_normalization_15/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_15_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_15/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_15/FusedBatchNormV3FusedBatchNormV3conv2d_16/BiasAdd:output:0-batch_normalization_15/ReadVariableOp:value:0/batch_normalization_15/ReadVariableOp_1:value:0>batch_normalization_15/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_15/FusedBatchNormV3?
%batch_normalization_15/AssignNewValueAssignVariableOp?batch_normalization_15_fusedbatchnormv3_readvariableop_resource4batch_normalization_15/FusedBatchNormV3:batch_mean:07^batch_normalization_15/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*R
_classH
FDloc:@batch_normalization_15/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_15/AssignNewValue?
'batch_normalization_15/AssignNewValue_1AssignVariableOpAbatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_15/FusedBatchNormV3:batch_variance:09^batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*T
_classJ
HFloc:@batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_15/AssignNewValue_1?
activation_15/ReluRelu+batch_normalization_15/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? 2
activation_15/Relu?
max_pooling2d_3/MaxPoolMaxPool activation_15/Relu:activations:0*
T0*1
_output_shapes
:??????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPoolu
dropout_6/dropout/ConstConst*
_output_shapes
: *
dtype0*
value
B j?x2
dropout_6/dropout/Const?
dropout_6/dropout/MulMul max_pooling2d_3/MaxPool:output:0 dropout_6/dropout/Const:output:0*
T0*1
_output_shapes
:??????????? 2
dropout_6/dropout/Mul?
dropout_6/dropout/ShapeShape max_pooling2d_3/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_6/dropout/Shape?
.dropout_6/dropout/random_uniform/RandomUniformRandomUniform dropout_6/dropout/Shape:output:0*
T0*1
_output_shapes
:??????????? *
dtype020
.dropout_6/dropout/random_uniform/RandomUniform?
 dropout_6/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value
B j?\2"
 dropout_6/dropout/GreaterEqual/y?
dropout_6/dropout/GreaterEqualGreaterEqual7dropout_6/dropout/random_uniform/RandomUniform:output:0)dropout_6/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:??????????? 2 
dropout_6/dropout/GreaterEqual?
dropout_6/dropout/CastCast"dropout_6/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:??????????? 2
dropout_6/dropout/Cast?
dropout_6/dropout/Mul_1Muldropout_6/dropout/Mul:z:0dropout_6/dropout/Cast:y:0*
T0*1
_output_shapes
:??????????? 2
dropout_6/dropout/Mul_1?
conv2d_18/Conv2D/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_18/Conv2D/ReadVariableOp?
conv2d_18/Conv2D/CastCast'conv2d_18/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: @2
conv2d_18/Conv2D/Cast?
conv2d_18/Conv2DConv2Ddropout_6/dropout/Mul_1:z:0conv2d_18/Conv2D/Cast:y:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv2d_18/Conv2D?
 conv2d_18/BiasAdd/ReadVariableOpReadVariableOp)conv2d_18_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_18/BiasAdd/ReadVariableOp?
conv2d_18/BiasAdd/CastCast(conv2d_18/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@2
conv2d_18/BiasAdd/Cast?
conv2d_18/BiasAddBiasAddconv2d_18/Conv2D:output:0conv2d_18/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:???????????@2
conv2d_18/BiasAdd?
%batch_normalization_17/ReadVariableOpReadVariableOp.batch_normalization_17_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_17/ReadVariableOp?
'batch_normalization_17/ReadVariableOp_1ReadVariableOp0batch_normalization_17_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_17/ReadVariableOp_1?
6batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_17/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_17/FusedBatchNormV3FusedBatchNormV3conv2d_18/BiasAdd:output:0-batch_normalization_17/ReadVariableOp:value:0/batch_normalization_17/ReadVariableOp_1:value:0>batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_17/FusedBatchNormV3?
%batch_normalization_17/AssignNewValueAssignVariableOp?batch_normalization_17_fusedbatchnormv3_readvariableop_resource4batch_normalization_17/FusedBatchNormV3:batch_mean:07^batch_normalization_17/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*R
_classH
FDloc:@batch_normalization_17/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_17/AssignNewValue?
'batch_normalization_17/AssignNewValue_1AssignVariableOpAbatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_17/FusedBatchNormV3:batch_variance:09^batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*T
_classJ
HFloc:@batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_17/AssignNewValue_1?
activation_17/ReluRelu+batch_normalization_17/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????@2
activation_17/Relu?
max_pooling2d_4/MaxPoolMaxPool activation_17/Relu:activations:0*
T0*1
_output_shapes
:???????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_4/MaxPoolu
dropout_7/dropout/ConstConst*
_output_shapes
: *
dtype0*
value
B j?x2
dropout_7/dropout/Const?
dropout_7/dropout/MulMul max_pooling2d_4/MaxPool:output:0 dropout_7/dropout/Const:output:0*
T0*1
_output_shapes
:???????????@2
dropout_7/dropout/Mul?
dropout_7/dropout/ShapeShape max_pooling2d_4/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_7/dropout/Shape?
.dropout_7/dropout/random_uniform/RandomUniformRandomUniform dropout_7/dropout/Shape:output:0*
T0*1
_output_shapes
:???????????@*
dtype020
.dropout_7/dropout/random_uniform/RandomUniform?
 dropout_7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value
B j?\2"
 dropout_7/dropout/GreaterEqual/y?
dropout_7/dropout/GreaterEqualGreaterEqual7dropout_7/dropout/random_uniform/RandomUniform:output:0)dropout_7/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:???????????@2 
dropout_7/dropout/GreaterEqual?
dropout_7/dropout/CastCast"dropout_7/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:???????????@2
dropout_7/dropout/Cast?
dropout_7/dropout/Mul_1Muldropout_7/dropout/Mul:z:0dropout_7/dropout/Cast:y:0*
T0*1
_output_shapes
:???????????@2
dropout_7/dropout/Mul_1?
conv2d_20/Conv2D/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02!
conv2d_20/Conv2D/ReadVariableOp?
conv2d_20/Conv2D/CastCast'conv2d_20/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*'
_output_shapes
:@?2
conv2d_20/Conv2D/Cast?
conv2d_20/Conv2DConv2Ddropout_7/dropout/Mul_1:z:0conv2d_20/Conv2D/Cast:y:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
conv2d_20/Conv2D?
 conv2d_20/BiasAdd/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_20/BiasAdd/ReadVariableOp?
conv2d_20/BiasAdd/CastCast(conv2d_20/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:?2
conv2d_20/BiasAdd/Cast?
conv2d_20/BiasAddBiasAddconv2d_20/Conv2D:output:0conv2d_20/BiasAdd/Cast:y:0*
T0*2
_output_shapes 
:????????????2
conv2d_20/BiasAdd?
%batch_normalization_19/ReadVariableOpReadVariableOp.batch_normalization_19_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_19/ReadVariableOp?
'batch_normalization_19/ReadVariableOp_1ReadVariableOp0batch_normalization_19_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_19/ReadVariableOp_1?
6batch_normalization_19/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_19_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_19/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_19/FusedBatchNormV3FusedBatchNormV3conv2d_20/BiasAdd:output:0-batch_normalization_19/ReadVariableOp:value:0/batch_normalization_19/ReadVariableOp_1:value:0>batch_normalization_19/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_19/FusedBatchNormV3?
%batch_normalization_19/AssignNewValueAssignVariableOp?batch_normalization_19_fusedbatchnormv3_readvariableop_resource4batch_normalization_19/FusedBatchNormV3:batch_mean:07^batch_normalization_19/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*R
_classH
FDloc:@batch_normalization_19/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_19/AssignNewValue?
'batch_normalization_19/AssignNewValue_1AssignVariableOpAbatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_19/FusedBatchNormV3:batch_variance:09^batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*T
_classJ
HFloc:@batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_19/AssignNewValue_1?
activation_19/ReluRelu+batch_normalization_19/FusedBatchNormV3:y:0*
T0*2
_output_shapes 
:????????????2
activation_19/Relu?
max_pooling2d_5/MaxPoolMaxPool activation_19/Relu:activations:0*
T0*0
_output_shapes
:?????????D\?*
ksize
*
paddingVALID*
strides
2
max_pooling2d_5/MaxPoolu
dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
value
B j?x2
dropout_8/dropout/Const?
dropout_8/dropout/MulMul max_pooling2d_5/MaxPool:output:0 dropout_8/dropout/Const:output:0*
T0*0
_output_shapes
:?????????D\?2
dropout_8/dropout/Mul?
dropout_8/dropout/ShapeShape max_pooling2d_5/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_8/dropout/Shape?
.dropout_8/dropout/random_uniform/RandomUniformRandomUniform dropout_8/dropout/Shape:output:0*
T0*0
_output_shapes
:?????????D\?*
dtype020
.dropout_8/dropout/random_uniform/RandomUniform?
 dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value
B j?\2"
 dropout_8/dropout/GreaterEqual/y?
dropout_8/dropout/GreaterEqualGreaterEqual7dropout_8/dropout/random_uniform/RandomUniform:output:0)dropout_8/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:?????????D\?2 
dropout_8/dropout/GreaterEqual?
dropout_8/dropout/CastCast"dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:?????????D\?2
dropout_8/dropout/Cast?
dropout_8/dropout/Mul_1Muldropout_8/dropout/Mul:z:0dropout_8/dropout/Cast:y:0*
T0*0
_output_shapes
:?????????D\?2
dropout_8/dropout/Mul_1?
conv2d_22/Conv2D/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_22/Conv2D/ReadVariableOp?
conv2d_22/Conv2D/CastCast'conv2d_22/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*(
_output_shapes
:??2
conv2d_22/Conv2D/Cast?
conv2d_22/Conv2DConv2Ddropout_8/dropout/Mul_1:z:0conv2d_22/Conv2D/Cast:y:0*
T0*0
_output_shapes
:?????????D\?*
paddingSAME*
strides
2
conv2d_22/Conv2D?
 conv2d_22/BiasAdd/ReadVariableOpReadVariableOp)conv2d_22_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_22/BiasAdd/ReadVariableOp?
conv2d_22/BiasAdd/CastCast(conv2d_22/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:?2
conv2d_22/BiasAdd/Cast?
conv2d_22/BiasAddBiasAddconv2d_22/Conv2D:output:0conv2d_22/BiasAdd/Cast:y:0*
T0*0
_output_shapes
:?????????D\?2
conv2d_22/BiasAdd?
%batch_normalization_21/ReadVariableOpReadVariableOp.batch_normalization_21_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_21/ReadVariableOp?
'batch_normalization_21/ReadVariableOp_1ReadVariableOp0batch_normalization_21_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_21/ReadVariableOp_1?
6batch_normalization_21/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_21_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_21/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_21/FusedBatchNormV3FusedBatchNormV3conv2d_22/BiasAdd:output:0-batch_normalization_21/ReadVariableOp:value:0/batch_normalization_21/ReadVariableOp_1:value:0>batch_normalization_21/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????D\?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_21/FusedBatchNormV3?
%batch_normalization_21/AssignNewValueAssignVariableOp?batch_normalization_21_fusedbatchnormv3_readvariableop_resource4batch_normalization_21/FusedBatchNormV3:batch_mean:07^batch_normalization_21/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*R
_classH
FDloc:@batch_normalization_21/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_21/AssignNewValue?
'batch_normalization_21/AssignNewValue_1AssignVariableOpAbatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_21/FusedBatchNormV3:batch_variance:09^batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*T
_classJ
HFloc:@batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_21/AssignNewValue_1?
activation_21/ReluRelu+batch_normalization_21/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????D\?2
activation_21/Relu?
conv2d_transpose_3/ShapeShape activation_21/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_3/Shape?
&conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_3/strided_slice/stack?
(conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_3/strided_slice/stack_1?
(conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_3/strided_slice/stack_2?
 conv2d_transpose_3/strided_sliceStridedSlice!conv2d_transpose_3/Shape:output:0/conv2d_transpose_3/strided_slice/stack:output:01conv2d_transpose_3/strided_slice/stack_1:output:01conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_3/strided_slice{
conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_3/stack/1{
conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_3/stack/2{
conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_3/stack/3?
conv2d_transpose_3/stackPack)conv2d_transpose_3/strided_slice:output:0#conv2d_transpose_3/stack/1:output:0#conv2d_transpose_3/stack/2:output:0#conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_3/stack?
(conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_3/strided_slice_1/stack?
*conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_1/stack_1?
*conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_1/stack_2?
"conv2d_transpose_3/strided_slice_1StridedSlice!conv2d_transpose_3/stack:output:01conv2d_transpose_3/strided_slice_1/stack:output:03conv2d_transpose_3/strided_slice_1/stack_1:output:03conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_3/strided_slice_1?
2conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_3_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype024
2conv2d_transpose_3/conv2d_transpose/ReadVariableOp?
(conv2d_transpose_3/conv2d_transpose/CastCast:conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0*

DstT0*

SrcT0*(
_output_shapes
:??2*
(conv2d_transpose_3/conv2d_transpose/Cast?
#conv2d_transpose_3/conv2d_transposeConv2DBackpropInput!conv2d_transpose_3/stack:output:0,conv2d_transpose_3/conv2d_transpose/Cast:y:0 activation_21/Relu:activations:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2%
#conv2d_transpose_3/conv2d_transpose?
)conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)conv2d_transpose_3/BiasAdd/ReadVariableOp?
conv2d_transpose_3/BiasAdd/CastCast1conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:?2!
conv2d_transpose_3/BiasAdd/Cast?
conv2d_transpose_3/BiasAddBiasAdd,conv2d_transpose_3/conv2d_transpose:output:0#conv2d_transpose_3/BiasAdd/Cast:y:0*
T0*2
_output_shapes 
:????????????2
conv2d_transpose_3/BiasAddx
concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_3/concat/axis?
concatenate_3/concatConcatV2#conv2d_transpose_3/BiasAdd:output:0 activation_19/Relu:activations:0"concatenate_3/concat/axis:output:0*
N*
T0*2
_output_shapes 
:????????????2
concatenate_3/concatu
dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
value
B j?x2
dropout_9/dropout/Const?
dropout_9/dropout/MulMulconcatenate_3/concat:output:0 dropout_9/dropout/Const:output:0*
T0*2
_output_shapes 
:????????????2
dropout_9/dropout/Mul
dropout_9/dropout/ShapeShapeconcatenate_3/concat:output:0*
T0*
_output_shapes
:2
dropout_9/dropout/Shape?
.dropout_9/dropout/random_uniform/RandomUniformRandomUniform dropout_9/dropout/Shape:output:0*
T0*2
_output_shapes 
:????????????*
dtype020
.dropout_9/dropout/random_uniform/RandomUniform?
 dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value
B j?\2"
 dropout_9/dropout/GreaterEqual/y?
dropout_9/dropout/GreaterEqualGreaterEqual7dropout_9/dropout/random_uniform/RandomUniform:output:0)dropout_9/dropout/GreaterEqual/y:output:0*
T0*2
_output_shapes 
:????????????2 
dropout_9/dropout/GreaterEqual?
dropout_9/dropout/CastCast"dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*2
_output_shapes 
:????????????2
dropout_9/dropout/Cast?
dropout_9/dropout/Mul_1Muldropout_9/dropout/Mul:z:0dropout_9/dropout/Cast:y:0*
T0*2
_output_shapes 
:????????????2
dropout_9/dropout/Mul_1?
conv2d_24/Conv2D/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_24/Conv2D/ReadVariableOp?
conv2d_24/Conv2D/CastCast'conv2d_24/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*(
_output_shapes
:??2
conv2d_24/Conv2D/Cast?
conv2d_24/Conv2DConv2Ddropout_9/dropout/Mul_1:z:0conv2d_24/Conv2D/Cast:y:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
conv2d_24/Conv2D?
 conv2d_24/BiasAdd/ReadVariableOpReadVariableOp)conv2d_24_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_24/BiasAdd/ReadVariableOp?
conv2d_24/BiasAdd/CastCast(conv2d_24/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:?2
conv2d_24/BiasAdd/Cast?
conv2d_24/BiasAddBiasAddconv2d_24/Conv2D:output:0conv2d_24/BiasAdd/Cast:y:0*
T0*2
_output_shapes 
:????????????2
conv2d_24/BiasAdd?
%batch_normalization_23/ReadVariableOpReadVariableOp.batch_normalization_23_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_23/ReadVariableOp?
'batch_normalization_23/ReadVariableOp_1ReadVariableOp0batch_normalization_23_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_23/ReadVariableOp_1?
6batch_normalization_23/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_23_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_23/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_23/FusedBatchNormV3FusedBatchNormV3conv2d_24/BiasAdd:output:0-batch_normalization_23/ReadVariableOp:value:0/batch_normalization_23/ReadVariableOp_1:value:0>batch_normalization_23/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_23/FusedBatchNormV3?
%batch_normalization_23/AssignNewValueAssignVariableOp?batch_normalization_23_fusedbatchnormv3_readvariableop_resource4batch_normalization_23/FusedBatchNormV3:batch_mean:07^batch_normalization_23/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*R
_classH
FDloc:@batch_normalization_23/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_23/AssignNewValue?
'batch_normalization_23/AssignNewValue_1AssignVariableOpAbatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_23/FusedBatchNormV3:batch_variance:09^batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*T
_classJ
HFloc:@batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_23/AssignNewValue_1?
activation_23/ReluRelu+batch_normalization_23/FusedBatchNormV3:y:0*
T0*2
_output_shapes 
:????????????2
activation_23/Relu?
conv2d_transpose_4/ShapeShape activation_23/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_4/Shape?
&conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_4/strided_slice/stack?
(conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_4/strided_slice/stack_1?
(conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_4/strided_slice/stack_2?
 conv2d_transpose_4/strided_sliceStridedSlice!conv2d_transpose_4/Shape:output:0/conv2d_transpose_4/strided_slice/stack:output:01conv2d_transpose_4/strided_slice/stack_1:output:01conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_4/strided_slice{
conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_4/stack/1{
conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_4/stack/2z
conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_4/stack/3?
conv2d_transpose_4/stackPack)conv2d_transpose_4/strided_slice:output:0#conv2d_transpose_4/stack/1:output:0#conv2d_transpose_4/stack/2:output:0#conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_4/stack?
(conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_4/strided_slice_1/stack?
*conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_4/strided_slice_1/stack_1?
*conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_4/strided_slice_1/stack_2?
"conv2d_transpose_4/strided_slice_1StridedSlice!conv2d_transpose_4/stack:output:01conv2d_transpose_4/strided_slice_1/stack:output:03conv2d_transpose_4/strided_slice_1/stack_1:output:03conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_4/strided_slice_1?
2conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_4_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype024
2conv2d_transpose_4/conv2d_transpose/ReadVariableOp?
(conv2d_transpose_4/conv2d_transpose/CastCast:conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0*

DstT0*

SrcT0*'
_output_shapes
:@?2*
(conv2d_transpose_4/conv2d_transpose/Cast?
#conv2d_transpose_4/conv2d_transposeConv2DBackpropInput!conv2d_transpose_4/stack:output:0,conv2d_transpose_4/conv2d_transpose/Cast:y:0 activation_23/Relu:activations:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2%
#conv2d_transpose_4/conv2d_transpose?
)conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)conv2d_transpose_4/BiasAdd/ReadVariableOp?
conv2d_transpose_4/BiasAdd/CastCast1conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@2!
conv2d_transpose_4/BiasAdd/Cast?
conv2d_transpose_4/BiasAddBiasAdd,conv2d_transpose_4/conv2d_transpose:output:0#conv2d_transpose_4/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:???????????@2
conv2d_transpose_4/BiasAddx
concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_4/concat/axis?
concatenate_4/concatConcatV2#conv2d_transpose_4/BiasAdd:output:0 activation_17/Relu:activations:0"concatenate_4/concat/axis:output:0*
N*
T0*2
_output_shapes 
:????????????2
concatenate_4/concatw
dropout_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
value
B j?x2
dropout_10/dropout/Const?
dropout_10/dropout/MulMulconcatenate_4/concat:output:0!dropout_10/dropout/Const:output:0*
T0*2
_output_shapes 
:????????????2
dropout_10/dropout/Mul?
dropout_10/dropout/ShapeShapeconcatenate_4/concat:output:0*
T0*
_output_shapes
:2
dropout_10/dropout/Shape?
/dropout_10/dropout/random_uniform/RandomUniformRandomUniform!dropout_10/dropout/Shape:output:0*
T0*2
_output_shapes 
:????????????*
dtype021
/dropout_10/dropout/random_uniform/RandomUniform?
!dropout_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value
B j?\2#
!dropout_10/dropout/GreaterEqual/y?
dropout_10/dropout/GreaterEqualGreaterEqual8dropout_10/dropout/random_uniform/RandomUniform:output:0*dropout_10/dropout/GreaterEqual/y:output:0*
T0*2
_output_shapes 
:????????????2!
dropout_10/dropout/GreaterEqual?
dropout_10/dropout/CastCast#dropout_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*2
_output_shapes 
:????????????2
dropout_10/dropout/Cast?
dropout_10/dropout/Mul_1Muldropout_10/dropout/Mul:z:0dropout_10/dropout/Cast:y:0*
T0*2
_output_shapes 
:????????????2
dropout_10/dropout/Mul_1?
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02!
conv2d_26/Conv2D/ReadVariableOp?
conv2d_26/Conv2D/CastCast'conv2d_26/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*'
_output_shapes
:?@2
conv2d_26/Conv2D/Cast?
conv2d_26/Conv2DConv2Ddropout_10/dropout/Mul_1:z:0conv2d_26/Conv2D/Cast:y:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv2d_26/Conv2D?
 conv2d_26/BiasAdd/ReadVariableOpReadVariableOp)conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_26/BiasAdd/ReadVariableOp?
conv2d_26/BiasAdd/CastCast(conv2d_26/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@2
conv2d_26/BiasAdd/Cast?
conv2d_26/BiasAddBiasAddconv2d_26/Conv2D:output:0conv2d_26/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:???????????@2
conv2d_26/BiasAdd?
%batch_normalization_25/ReadVariableOpReadVariableOp.batch_normalization_25_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_25/ReadVariableOp?
'batch_normalization_25/ReadVariableOp_1ReadVariableOp0batch_normalization_25_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_25/ReadVariableOp_1?
6batch_normalization_25/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_25_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_25/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_25_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_25/FusedBatchNormV3FusedBatchNormV3conv2d_26/BiasAdd:output:0-batch_normalization_25/ReadVariableOp:value:0/batch_normalization_25/ReadVariableOp_1:value:0>batch_normalization_25/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_25/FusedBatchNormV3?
%batch_normalization_25/AssignNewValueAssignVariableOp?batch_normalization_25_fusedbatchnormv3_readvariableop_resource4batch_normalization_25/FusedBatchNormV3:batch_mean:07^batch_normalization_25/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*R
_classH
FDloc:@batch_normalization_25/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_25/AssignNewValue?
'batch_normalization_25/AssignNewValue_1AssignVariableOpAbatch_normalization_25_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_25/FusedBatchNormV3:batch_variance:09^batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*T
_classJ
HFloc:@batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_25/AssignNewValue_1?
activation_25/ReluRelu+batch_normalization_25/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????@2
activation_25/Relu?
conv2d_transpose_5/ShapeShape activation_25/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_5/Shape?
&conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_5/strided_slice/stack?
(conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_5/strided_slice/stack_1?
(conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_5/strided_slice/stack_2?
 conv2d_transpose_5/strided_sliceStridedSlice!conv2d_transpose_5/Shape:output:0/conv2d_transpose_5/strided_slice/stack:output:01conv2d_transpose_5/strided_slice/stack_1:output:01conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_5/strided_slice{
conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_5/stack/1{
conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_5/stack/2z
conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_5/stack/3?
conv2d_transpose_5/stackPack)conv2d_transpose_5/strided_slice:output:0#conv2d_transpose_5/stack/1:output:0#conv2d_transpose_5/stack/2:output:0#conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_5/stack?
(conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_5/strided_slice_1/stack?
*conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_5/strided_slice_1/stack_1?
*conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_5/strided_slice_1/stack_2?
"conv2d_transpose_5/strided_slice_1StridedSlice!conv2d_transpose_5/stack:output:01conv2d_transpose_5/strided_slice_1/stack:output:03conv2d_transpose_5/strided_slice_1/stack_1:output:03conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_5/strided_slice_1?
2conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype024
2conv2d_transpose_5/conv2d_transpose/ReadVariableOp?
(conv2d_transpose_5/conv2d_transpose/CastCast:conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: @2*
(conv2d_transpose_5/conv2d_transpose/Cast?
#conv2d_transpose_5/conv2d_transposeConv2DBackpropInput!conv2d_transpose_5/stack:output:0,conv2d_transpose_5/conv2d_transpose/Cast:y:0 activation_25/Relu:activations:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2%
#conv2d_transpose_5/conv2d_transpose?
)conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)conv2d_transpose_5/BiasAdd/ReadVariableOp?
conv2d_transpose_5/BiasAdd/CastCast1conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: 2!
conv2d_transpose_5/BiasAdd/Cast?
conv2d_transpose_5/BiasAddBiasAdd,conv2d_transpose_5/conv2d_transpose:output:0#conv2d_transpose_5/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:??????????? 2
conv2d_transpose_5/BiasAddx
concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_5/concat/axis?
concatenate_5/concatConcatV2#conv2d_transpose_5/BiasAdd:output:0 activation_15/Relu:activations:0"concatenate_5/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????@2
concatenate_5/concatw
dropout_11/dropout/ConstConst*
_output_shapes
: *
dtype0*
value
B j?x2
dropout_11/dropout/Const?
dropout_11/dropout/MulMulconcatenate_5/concat:output:0!dropout_11/dropout/Const:output:0*
T0*1
_output_shapes
:???????????@2
dropout_11/dropout/Mul?
dropout_11/dropout/ShapeShapeconcatenate_5/concat:output:0*
T0*
_output_shapes
:2
dropout_11/dropout/Shape?
/dropout_11/dropout/random_uniform/RandomUniformRandomUniform!dropout_11/dropout/Shape:output:0*
T0*1
_output_shapes
:???????????@*
dtype021
/dropout_11/dropout/random_uniform/RandomUniform?
!dropout_11/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value
B j?\2#
!dropout_11/dropout/GreaterEqual/y?
dropout_11/dropout/GreaterEqualGreaterEqual8dropout_11/dropout/random_uniform/RandomUniform:output:0*dropout_11/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:???????????@2!
dropout_11/dropout/GreaterEqual?
dropout_11/dropout/CastCast#dropout_11/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:???????????@2
dropout_11/dropout/Cast?
dropout_11/dropout/Mul_1Muldropout_11/dropout/Mul:z:0dropout_11/dropout/Cast:y:0*
T0*1
_output_shapes
:???????????@2
dropout_11/dropout/Mul_1?
conv2d_28/Conv2D/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02!
conv2d_28/Conv2D/ReadVariableOp?
conv2d_28/Conv2D/CastCast'conv2d_28/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:@ 2
conv2d_28/Conv2D/Cast?
conv2d_28/Conv2DConv2Ddropout_11/dropout/Mul_1:z:0conv2d_28/Conv2D/Cast:y:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
conv2d_28/Conv2D?
 conv2d_28/BiasAdd/ReadVariableOpReadVariableOp)conv2d_28_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_28/BiasAdd/ReadVariableOp?
conv2d_28/BiasAdd/CastCast(conv2d_28/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: 2
conv2d_28/BiasAdd/Cast?
conv2d_28/BiasAddBiasAddconv2d_28/Conv2D:output:0conv2d_28/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:??????????? 2
conv2d_28/BiasAdd?
%batch_normalization_27/ReadVariableOpReadVariableOp.batch_normalization_27_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_27/ReadVariableOp?
'batch_normalization_27/ReadVariableOp_1ReadVariableOp0batch_normalization_27_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_27/ReadVariableOp_1?
6batch_normalization_27/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_27_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_27/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_27/FusedBatchNormV3FusedBatchNormV3conv2d_28/BiasAdd:output:0-batch_normalization_27/ReadVariableOp:value:0/batch_normalization_27/ReadVariableOp_1:value:0>batch_normalization_27/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_27/FusedBatchNormV3?
%batch_normalization_27/AssignNewValueAssignVariableOp?batch_normalization_27_fusedbatchnormv3_readvariableop_resource4batch_normalization_27/FusedBatchNormV3:batch_mean:07^batch_normalization_27/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*R
_classH
FDloc:@batch_normalization_27/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_27/AssignNewValue?
'batch_normalization_27/AssignNewValue_1AssignVariableOpAbatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_27/FusedBatchNormV3:batch_variance:09^batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*T
_classJ
HFloc:@batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_27/AssignNewValue_1?
activation_27/ReluRelu+batch_normalization_27/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? 2
activation_27/Relu?
conv2d_29/Conv2D/ReadVariableOpReadVariableOp(conv2d_29_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_29/Conv2D/ReadVariableOp?
conv2d_29/Conv2D/CastCast'conv2d_29/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: 2
conv2d_29/Conv2D/Cast?
conv2d_29/Conv2DConv2D activation_27/Relu:activations:0conv2d_29/Conv2D/Cast:y:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
conv2d_29/Conv2D?
 conv2d_29/BiasAdd/ReadVariableOpReadVariableOp)conv2d_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_29/BiasAdd/ReadVariableOp?
conv2d_29/BiasAdd/CastCast(conv2d_29/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:2
conv2d_29/BiasAdd/Cast?
conv2d_29/BiasAddBiasAddconv2d_29/Conv2D:output:0conv2d_29/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:???????????2
conv2d_29/BiasAdd?
conv2d_29/SigmoidSigmoidconv2d_29/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_29/Sigmoid?
IdentityIdentityconv2d_29/Sigmoid:y:0&^batch_normalization_15/AssignNewValue(^batch_normalization_15/AssignNewValue_17^batch_normalization_15/FusedBatchNormV3/ReadVariableOp9^batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_15/ReadVariableOp(^batch_normalization_15/ReadVariableOp_1&^batch_normalization_17/AssignNewValue(^batch_normalization_17/AssignNewValue_17^batch_normalization_17/FusedBatchNormV3/ReadVariableOp9^batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_17/ReadVariableOp(^batch_normalization_17/ReadVariableOp_1&^batch_normalization_19/AssignNewValue(^batch_normalization_19/AssignNewValue_17^batch_normalization_19/FusedBatchNormV3/ReadVariableOp9^batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_19/ReadVariableOp(^batch_normalization_19/ReadVariableOp_1&^batch_normalization_21/AssignNewValue(^batch_normalization_21/AssignNewValue_17^batch_normalization_21/FusedBatchNormV3/ReadVariableOp9^batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_21/ReadVariableOp(^batch_normalization_21/ReadVariableOp_1&^batch_normalization_23/AssignNewValue(^batch_normalization_23/AssignNewValue_17^batch_normalization_23/FusedBatchNormV3/ReadVariableOp9^batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_23/ReadVariableOp(^batch_normalization_23/ReadVariableOp_1&^batch_normalization_25/AssignNewValue(^batch_normalization_25/AssignNewValue_17^batch_normalization_25/FusedBatchNormV3/ReadVariableOp9^batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_25/ReadVariableOp(^batch_normalization_25/ReadVariableOp_1&^batch_normalization_27/AssignNewValue(^batch_normalization_27/AssignNewValue_17^batch_normalization_27/FusedBatchNormV3/ReadVariableOp9^batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_27/ReadVariableOp(^batch_normalization_27/ReadVariableOp_1!^conv2d_16/BiasAdd/ReadVariableOp ^conv2d_16/Conv2D/ReadVariableOp!^conv2d_18/BiasAdd/ReadVariableOp ^conv2d_18/Conv2D/ReadVariableOp!^conv2d_20/BiasAdd/ReadVariableOp ^conv2d_20/Conv2D/ReadVariableOp!^conv2d_22/BiasAdd/ReadVariableOp ^conv2d_22/Conv2D/ReadVariableOp!^conv2d_24/BiasAdd/ReadVariableOp ^conv2d_24/Conv2D/ReadVariableOp!^conv2d_26/BiasAdd/ReadVariableOp ^conv2d_26/Conv2D/ReadVariableOp!^conv2d_28/BiasAdd/ReadVariableOp ^conv2d_28/Conv2D/ReadVariableOp!^conv2d_29/BiasAdd/ReadVariableOp ^conv2d_29/Conv2D/ReadVariableOp*^conv2d_transpose_3/BiasAdd/ReadVariableOp3^conv2d_transpose_3/conv2d_transpose/ReadVariableOp*^conv2d_transpose_4/BiasAdd/ReadVariableOp3^conv2d_transpose_4/conv2d_transpose/ReadVariableOp*^conv2d_transpose_5/BiasAdd/ReadVariableOp3^conv2d_transpose_5/conv2d_transpose/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::::::::::2N
%batch_normalization_15/AssignNewValue%batch_normalization_15/AssignNewValue2R
'batch_normalization_15/AssignNewValue_1'batch_normalization_15/AssignNewValue_12p
6batch_normalization_15/FusedBatchNormV3/ReadVariableOp6batch_normalization_15/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_18batch_normalization_15/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_15/ReadVariableOp%batch_normalization_15/ReadVariableOp2R
'batch_normalization_15/ReadVariableOp_1'batch_normalization_15/ReadVariableOp_12N
%batch_normalization_17/AssignNewValue%batch_normalization_17/AssignNewValue2R
'batch_normalization_17/AssignNewValue_1'batch_normalization_17/AssignNewValue_12p
6batch_normalization_17/FusedBatchNormV3/ReadVariableOp6batch_normalization_17/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_18batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_17/ReadVariableOp%batch_normalization_17/ReadVariableOp2R
'batch_normalization_17/ReadVariableOp_1'batch_normalization_17/ReadVariableOp_12N
%batch_normalization_19/AssignNewValue%batch_normalization_19/AssignNewValue2R
'batch_normalization_19/AssignNewValue_1'batch_normalization_19/AssignNewValue_12p
6batch_normalization_19/FusedBatchNormV3/ReadVariableOp6batch_normalization_19/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_18batch_normalization_19/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_19/ReadVariableOp%batch_normalization_19/ReadVariableOp2R
'batch_normalization_19/ReadVariableOp_1'batch_normalization_19/ReadVariableOp_12N
%batch_normalization_21/AssignNewValue%batch_normalization_21/AssignNewValue2R
'batch_normalization_21/AssignNewValue_1'batch_normalization_21/AssignNewValue_12p
6batch_normalization_21/FusedBatchNormV3/ReadVariableOp6batch_normalization_21/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_18batch_normalization_21/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_21/ReadVariableOp%batch_normalization_21/ReadVariableOp2R
'batch_normalization_21/ReadVariableOp_1'batch_normalization_21/ReadVariableOp_12N
%batch_normalization_23/AssignNewValue%batch_normalization_23/AssignNewValue2R
'batch_normalization_23/AssignNewValue_1'batch_normalization_23/AssignNewValue_12p
6batch_normalization_23/FusedBatchNormV3/ReadVariableOp6batch_normalization_23/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_18batch_normalization_23/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_23/ReadVariableOp%batch_normalization_23/ReadVariableOp2R
'batch_normalization_23/ReadVariableOp_1'batch_normalization_23/ReadVariableOp_12N
%batch_normalization_25/AssignNewValue%batch_normalization_25/AssignNewValue2R
'batch_normalization_25/AssignNewValue_1'batch_normalization_25/AssignNewValue_12p
6batch_normalization_25/FusedBatchNormV3/ReadVariableOp6batch_normalization_25/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_18batch_normalization_25/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_25/ReadVariableOp%batch_normalization_25/ReadVariableOp2R
'batch_normalization_25/ReadVariableOp_1'batch_normalization_25/ReadVariableOp_12N
%batch_normalization_27/AssignNewValue%batch_normalization_27/AssignNewValue2R
'batch_normalization_27/AssignNewValue_1'batch_normalization_27/AssignNewValue_12p
6batch_normalization_27/FusedBatchNormV3/ReadVariableOp6batch_normalization_27/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_18batch_normalization_27/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_27/ReadVariableOp%batch_normalization_27/ReadVariableOp2R
'batch_normalization_27/ReadVariableOp_1'batch_normalization_27/ReadVariableOp_12D
 conv2d_16/BiasAdd/ReadVariableOp conv2d_16/BiasAdd/ReadVariableOp2B
conv2d_16/Conv2D/ReadVariableOpconv2d_16/Conv2D/ReadVariableOp2D
 conv2d_18/BiasAdd/ReadVariableOp conv2d_18/BiasAdd/ReadVariableOp2B
conv2d_18/Conv2D/ReadVariableOpconv2d_18/Conv2D/ReadVariableOp2D
 conv2d_20/BiasAdd/ReadVariableOp conv2d_20/BiasAdd/ReadVariableOp2B
conv2d_20/Conv2D/ReadVariableOpconv2d_20/Conv2D/ReadVariableOp2D
 conv2d_22/BiasAdd/ReadVariableOp conv2d_22/BiasAdd/ReadVariableOp2B
conv2d_22/Conv2D/ReadVariableOpconv2d_22/Conv2D/ReadVariableOp2D
 conv2d_24/BiasAdd/ReadVariableOp conv2d_24/BiasAdd/ReadVariableOp2B
conv2d_24/Conv2D/ReadVariableOpconv2d_24/Conv2D/ReadVariableOp2D
 conv2d_26/BiasAdd/ReadVariableOp conv2d_26/BiasAdd/ReadVariableOp2B
conv2d_26/Conv2D/ReadVariableOpconv2d_26/Conv2D/ReadVariableOp2D
 conv2d_28/BiasAdd/ReadVariableOp conv2d_28/BiasAdd/ReadVariableOp2B
conv2d_28/Conv2D/ReadVariableOpconv2d_28/Conv2D/ReadVariableOp2D
 conv2d_29/BiasAdd/ReadVariableOp conv2d_29/BiasAdd/ReadVariableOp2B
conv2d_29/Conv2D/ReadVariableOpconv2d_29/Conv2D/ReadVariableOp2V
)conv2d_transpose_3/BiasAdd/ReadVariableOp)conv2d_transpose_3/BiasAdd/ReadVariableOp2h
2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_4/BiasAdd/ReadVariableOp)conv2d_transpose_4/BiasAdd/ReadVariableOp2h
2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_5/BiasAdd/ReadVariableOp)conv2d_transpose_5/BiasAdd/ReadVariableOp2h
2conv2d_transpose_5/conv2d_transpose/ReadVariableOp2conv2d_transpose_5/conv2d_transpose/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_15_layer_call_fn_41107

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_375922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_37756

inputs
identity?
MaxPoolMaxPoolinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_21_layer_call_and_return_conditional_losses_41652

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
b
D__inference_dropout_8_layer_call_and_return_conditional_losses_38859

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:?????????D\?2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:?????????D\?2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:?????????D\?:X T
0
_output_shapes
:?????????D\?
 
_user_specified_nameinputs
?%
?
M__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_38168

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transpose/CastCast'conv2d_transpose/ReadVariableOp:value:0*

DstT0*

SrcT0*'
_output_shapes
:@?2
conv2d_transpose/Cast?
conv2d_transposeConv2DBackpropInputstack:output:0conv2d_transpose/Cast:y:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpx
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@2
BiasAdd/Cast?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/Cast:y:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_25_layer_call_and_return_conditional_losses_38271

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
~
)__inference_conv2d_28_layer_call_fn_42211

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_conv2d_28_layer_call_and_return_conditional_losses_393792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????@::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_27_layer_call_fn_42326

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_27_layer_call_and_return_conditional_losses_383902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
t
H__inference_concatenate_4_layer_call_and_return_conditional_losses_41958
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*2
_output_shapes 
:????????????2
concatn
IdentityIdentityconcat:output:0*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:+???????????????????????????@:???????????@:k g
A
_output_shapes/
-:+???????????????????????????@
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:???????????@
"
_user_specified_name
inputs/1
?
?
D__inference_conv2d_29_layer_call_and_return_conditional_losses_42362

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: 2
Conv2D/Cast?
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpx
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:2
BiasAdd/Cast|
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*1
_output_shapes
:???????????2	
BiasAddk
SigmoidSigmoidBiasAdd:output:0*
T0*1
_output_shapes
:???????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:??????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
E
)__inference_dropout_7_layer_call_fn_41407

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_387142
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????@:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
b
)__inference_dropout_6_layer_call_fn_41216

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_385642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*0
_input_shapes
:??????????? 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
t
H__inference_concatenate_5_layer_call_and_return_conditional_losses_42157
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????@2
concatm
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:+??????????????????????????? :??????????? :k g
A
_output_shapes/
-:+??????????????????????????? 
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:??????????? 
"
_user_specified_name
inputs/1
?
?
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_41530

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_38629

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_25_layer_call_and_return_conditional_losses_42114

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
d
E__inference_dropout_11_layer_call_and_return_conditional_losses_39349

inputs
identity?a
dropout/ConstConst*
_output_shapes
: *
dtype0*
value
B j?x2
dropout/Const}
dropout/MulMulinputsdropout/Const:output:0*
T0*1
_output_shapes
:???????????@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*1
_output_shapes
:???????????@*
dtype02&
$dropout/random_uniform/RandomUniforms
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value
B j?\2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:???????????@2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:???????????@2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*1
_output_shapes
:???????????@2
dropout/Mul_1o
IdentityIdentitydropout/Mul_1:z:0*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????@:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_37824

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_15_layer_call_fn_41120

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_376232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_15_layer_call_fn_41184

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_385022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:??????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
d
H__inference_activation_21_layer_call_and_return_conditional_losses_38978

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????D\?2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????D\?2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????D\?:X T
0
_output_shapes
:?????????D\?
 
_user_specified_nameinputs
??
?.
 __inference__wrapped_model_37530
input_24
0model_1_conv2d_16_conv2d_readvariableop_resource5
1model_1_conv2d_16_biasadd_readvariableop_resource:
6model_1_batch_normalization_15_readvariableop_resource<
8model_1_batch_normalization_15_readvariableop_1_resourceK
Gmodel_1_batch_normalization_15_fusedbatchnormv3_readvariableop_resourceM
Imodel_1_batch_normalization_15_fusedbatchnormv3_readvariableop_1_resource4
0model_1_conv2d_18_conv2d_readvariableop_resource5
1model_1_conv2d_18_biasadd_readvariableop_resource:
6model_1_batch_normalization_17_readvariableop_resource<
8model_1_batch_normalization_17_readvariableop_1_resourceK
Gmodel_1_batch_normalization_17_fusedbatchnormv3_readvariableop_resourceM
Imodel_1_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource4
0model_1_conv2d_20_conv2d_readvariableop_resource5
1model_1_conv2d_20_biasadd_readvariableop_resource:
6model_1_batch_normalization_19_readvariableop_resource<
8model_1_batch_normalization_19_readvariableop_1_resourceK
Gmodel_1_batch_normalization_19_fusedbatchnormv3_readvariableop_resourceM
Imodel_1_batch_normalization_19_fusedbatchnormv3_readvariableop_1_resource4
0model_1_conv2d_22_conv2d_readvariableop_resource5
1model_1_conv2d_22_biasadd_readvariableop_resource:
6model_1_batch_normalization_21_readvariableop_resource<
8model_1_batch_normalization_21_readvariableop_1_resourceK
Gmodel_1_batch_normalization_21_fusedbatchnormv3_readvariableop_resourceM
Imodel_1_batch_normalization_21_fusedbatchnormv3_readvariableop_1_resourceG
Cmodel_1_conv2d_transpose_3_conv2d_transpose_readvariableop_resource>
:model_1_conv2d_transpose_3_biasadd_readvariableop_resource4
0model_1_conv2d_24_conv2d_readvariableop_resource5
1model_1_conv2d_24_biasadd_readvariableop_resource:
6model_1_batch_normalization_23_readvariableop_resource<
8model_1_batch_normalization_23_readvariableop_1_resourceK
Gmodel_1_batch_normalization_23_fusedbatchnormv3_readvariableop_resourceM
Imodel_1_batch_normalization_23_fusedbatchnormv3_readvariableop_1_resourceG
Cmodel_1_conv2d_transpose_4_conv2d_transpose_readvariableop_resource>
:model_1_conv2d_transpose_4_biasadd_readvariableop_resource4
0model_1_conv2d_26_conv2d_readvariableop_resource5
1model_1_conv2d_26_biasadd_readvariableop_resource:
6model_1_batch_normalization_25_readvariableop_resource<
8model_1_batch_normalization_25_readvariableop_1_resourceK
Gmodel_1_batch_normalization_25_fusedbatchnormv3_readvariableop_resourceM
Imodel_1_batch_normalization_25_fusedbatchnormv3_readvariableop_1_resourceG
Cmodel_1_conv2d_transpose_5_conv2d_transpose_readvariableop_resource>
:model_1_conv2d_transpose_5_biasadd_readvariableop_resource4
0model_1_conv2d_28_conv2d_readvariableop_resource5
1model_1_conv2d_28_biasadd_readvariableop_resource:
6model_1_batch_normalization_27_readvariableop_resource<
8model_1_batch_normalization_27_readvariableop_1_resourceK
Gmodel_1_batch_normalization_27_fusedbatchnormv3_readvariableop_resourceM
Imodel_1_batch_normalization_27_fusedbatchnormv3_readvariableop_1_resource4
0model_1_conv2d_29_conv2d_readvariableop_resource5
1model_1_conv2d_29_biasadd_readvariableop_resource
identity??>model_1/batch_normalization_15/FusedBatchNormV3/ReadVariableOp?@model_1/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1?-model_1/batch_normalization_15/ReadVariableOp?/model_1/batch_normalization_15/ReadVariableOp_1?>model_1/batch_normalization_17/FusedBatchNormV3/ReadVariableOp?@model_1/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1?-model_1/batch_normalization_17/ReadVariableOp?/model_1/batch_normalization_17/ReadVariableOp_1?>model_1/batch_normalization_19/FusedBatchNormV3/ReadVariableOp?@model_1/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1?-model_1/batch_normalization_19/ReadVariableOp?/model_1/batch_normalization_19/ReadVariableOp_1?>model_1/batch_normalization_21/FusedBatchNormV3/ReadVariableOp?@model_1/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1?-model_1/batch_normalization_21/ReadVariableOp?/model_1/batch_normalization_21/ReadVariableOp_1?>model_1/batch_normalization_23/FusedBatchNormV3/ReadVariableOp?@model_1/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1?-model_1/batch_normalization_23/ReadVariableOp?/model_1/batch_normalization_23/ReadVariableOp_1?>model_1/batch_normalization_25/FusedBatchNormV3/ReadVariableOp?@model_1/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1?-model_1/batch_normalization_25/ReadVariableOp?/model_1/batch_normalization_25/ReadVariableOp_1?>model_1/batch_normalization_27/FusedBatchNormV3/ReadVariableOp?@model_1/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1?-model_1/batch_normalization_27/ReadVariableOp?/model_1/batch_normalization_27/ReadVariableOp_1?(model_1/conv2d_16/BiasAdd/ReadVariableOp?'model_1/conv2d_16/Conv2D/ReadVariableOp?(model_1/conv2d_18/BiasAdd/ReadVariableOp?'model_1/conv2d_18/Conv2D/ReadVariableOp?(model_1/conv2d_20/BiasAdd/ReadVariableOp?'model_1/conv2d_20/Conv2D/ReadVariableOp?(model_1/conv2d_22/BiasAdd/ReadVariableOp?'model_1/conv2d_22/Conv2D/ReadVariableOp?(model_1/conv2d_24/BiasAdd/ReadVariableOp?'model_1/conv2d_24/Conv2D/ReadVariableOp?(model_1/conv2d_26/BiasAdd/ReadVariableOp?'model_1/conv2d_26/Conv2D/ReadVariableOp?(model_1/conv2d_28/BiasAdd/ReadVariableOp?'model_1/conv2d_28/Conv2D/ReadVariableOp?(model_1/conv2d_29/BiasAdd/ReadVariableOp?'model_1/conv2d_29/Conv2D/ReadVariableOp?1model_1/conv2d_transpose_3/BiasAdd/ReadVariableOp?:model_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOp?1model_1/conv2d_transpose_4/BiasAdd/ReadVariableOp?:model_1/conv2d_transpose_4/conv2d_transpose/ReadVariableOp?1model_1/conv2d_transpose_5/BiasAdd/ReadVariableOp?:model_1/conv2d_transpose_5/conv2d_transpose/ReadVariableOpx
model_1/CastCastinput_2*

DstT0*

SrcT0*1
_output_shapes
:???????????2
model_1/Cast?
'model_1/conv2d_16/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'model_1/conv2d_16/Conv2D/ReadVariableOp?
model_1/conv2d_16/Conv2D/CastCast/model_1/conv2d_16/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: 2
model_1/conv2d_16/Conv2D/Cast?
model_1/conv2d_16/Conv2DConv2Dmodel_1/Cast:y:0!model_1/conv2d_16/Conv2D/Cast:y:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
model_1/conv2d_16/Conv2D?
(model_1/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model_1/conv2d_16/BiasAdd/ReadVariableOp?
model_1/conv2d_16/BiasAdd/CastCast0model_1/conv2d_16/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: 2 
model_1/conv2d_16/BiasAdd/Cast?
model_1/conv2d_16/BiasAddBiasAdd!model_1/conv2d_16/Conv2D:output:0"model_1/conv2d_16/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:??????????? 2
model_1/conv2d_16/BiasAdd?
-model_1/batch_normalization_15/ReadVariableOpReadVariableOp6model_1_batch_normalization_15_readvariableop_resource*
_output_shapes
: *
dtype02/
-model_1/batch_normalization_15/ReadVariableOp?
/model_1/batch_normalization_15/ReadVariableOp_1ReadVariableOp8model_1_batch_normalization_15_readvariableop_1_resource*
_output_shapes
: *
dtype021
/model_1/batch_normalization_15/ReadVariableOp_1?
>model_1/batch_normalization_15/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_1_batch_normalization_15_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02@
>model_1/batch_normalization_15/FusedBatchNormV3/ReadVariableOp?
@model_1/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_1_batch_normalization_15_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02B
@model_1/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1?
/model_1/batch_normalization_15/FusedBatchNormV3FusedBatchNormV3"model_1/conv2d_16/BiasAdd:output:05model_1/batch_normalization_15/ReadVariableOp:value:07model_1/batch_normalization_15/ReadVariableOp_1:value:0Fmodel_1/batch_normalization_15/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_1/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 21
/model_1/batch_normalization_15/FusedBatchNormV3?
model_1/activation_15/ReluRelu3model_1/batch_normalization_15/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? 2
model_1/activation_15/Relu?
model_1/max_pooling2d_3/MaxPoolMaxPool(model_1/activation_15/Relu:activations:0*
T0*1
_output_shapes
:??????????? *
ksize
*
paddingVALID*
strides
2!
model_1/max_pooling2d_3/MaxPool?
model_1/dropout_6/IdentityIdentity(model_1/max_pooling2d_3/MaxPool:output:0*
T0*1
_output_shapes
:??????????? 2
model_1/dropout_6/Identity?
'model_1/conv2d_18/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02)
'model_1/conv2d_18/Conv2D/ReadVariableOp?
model_1/conv2d_18/Conv2D/CastCast/model_1/conv2d_18/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: @2
model_1/conv2d_18/Conv2D/Cast?
model_1/conv2d_18/Conv2DConv2D#model_1/dropout_6/Identity:output:0!model_1/conv2d_18/Conv2D/Cast:y:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
model_1/conv2d_18/Conv2D?
(model_1/conv2d_18/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_18_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_1/conv2d_18/BiasAdd/ReadVariableOp?
model_1/conv2d_18/BiasAdd/CastCast0model_1/conv2d_18/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@2 
model_1/conv2d_18/BiasAdd/Cast?
model_1/conv2d_18/BiasAddBiasAdd!model_1/conv2d_18/Conv2D:output:0"model_1/conv2d_18/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:???????????@2
model_1/conv2d_18/BiasAdd?
-model_1/batch_normalization_17/ReadVariableOpReadVariableOp6model_1_batch_normalization_17_readvariableop_resource*
_output_shapes
:@*
dtype02/
-model_1/batch_normalization_17/ReadVariableOp?
/model_1/batch_normalization_17/ReadVariableOp_1ReadVariableOp8model_1_batch_normalization_17_readvariableop_1_resource*
_output_shapes
:@*
dtype021
/model_1/batch_normalization_17/ReadVariableOp_1?
>model_1/batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_1_batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02@
>model_1/batch_normalization_17/FusedBatchNormV3/ReadVariableOp?
@model_1/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_1_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02B
@model_1/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1?
/model_1/batch_normalization_17/FusedBatchNormV3FusedBatchNormV3"model_1/conv2d_18/BiasAdd:output:05model_1/batch_normalization_17/ReadVariableOp:value:07model_1/batch_normalization_17/ReadVariableOp_1:value:0Fmodel_1/batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_1/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
is_training( 21
/model_1/batch_normalization_17/FusedBatchNormV3?
model_1/activation_17/ReluRelu3model_1/batch_normalization_17/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????@2
model_1/activation_17/Relu?
model_1/max_pooling2d_4/MaxPoolMaxPool(model_1/activation_17/Relu:activations:0*
T0*1
_output_shapes
:???????????@*
ksize
*
paddingVALID*
strides
2!
model_1/max_pooling2d_4/MaxPool?
model_1/dropout_7/IdentityIdentity(model_1/max_pooling2d_4/MaxPool:output:0*
T0*1
_output_shapes
:???????????@2
model_1/dropout_7/Identity?
'model_1/conv2d_20/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_20_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02)
'model_1/conv2d_20/Conv2D/ReadVariableOp?
model_1/conv2d_20/Conv2D/CastCast/model_1/conv2d_20/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*'
_output_shapes
:@?2
model_1/conv2d_20/Conv2D/Cast?
model_1/conv2d_20/Conv2DConv2D#model_1/dropout_7/Identity:output:0!model_1/conv2d_20/Conv2D/Cast:y:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
model_1/conv2d_20/Conv2D?
(model_1/conv2d_20/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_20_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(model_1/conv2d_20/BiasAdd/ReadVariableOp?
model_1/conv2d_20/BiasAdd/CastCast0model_1/conv2d_20/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:?2 
model_1/conv2d_20/BiasAdd/Cast?
model_1/conv2d_20/BiasAddBiasAdd!model_1/conv2d_20/Conv2D:output:0"model_1/conv2d_20/BiasAdd/Cast:y:0*
T0*2
_output_shapes 
:????????????2
model_1/conv2d_20/BiasAdd?
-model_1/batch_normalization_19/ReadVariableOpReadVariableOp6model_1_batch_normalization_19_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-model_1/batch_normalization_19/ReadVariableOp?
/model_1/batch_normalization_19/ReadVariableOp_1ReadVariableOp8model_1_batch_normalization_19_readvariableop_1_resource*
_output_shapes	
:?*
dtype021
/model_1/batch_normalization_19/ReadVariableOp_1?
>model_1/batch_normalization_19/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_1_batch_normalization_19_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02@
>model_1/batch_normalization_19/FusedBatchNormV3/ReadVariableOp?
@model_1/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_1_batch_normalization_19_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02B
@model_1/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1?
/model_1/batch_normalization_19/FusedBatchNormV3FusedBatchNormV3"model_1/conv2d_20/BiasAdd:output:05model_1/batch_normalization_19/ReadVariableOp:value:07model_1/batch_normalization_19/ReadVariableOp_1:value:0Fmodel_1/batch_normalization_19/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_1/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:????????????:?:?:?:?:*
epsilon%o?:*
is_training( 21
/model_1/batch_normalization_19/FusedBatchNormV3?
model_1/activation_19/ReluRelu3model_1/batch_normalization_19/FusedBatchNormV3:y:0*
T0*2
_output_shapes 
:????????????2
model_1/activation_19/Relu?
model_1/max_pooling2d_5/MaxPoolMaxPool(model_1/activation_19/Relu:activations:0*
T0*0
_output_shapes
:?????????D\?*
ksize
*
paddingVALID*
strides
2!
model_1/max_pooling2d_5/MaxPool?
model_1/dropout_8/IdentityIdentity(model_1/max_pooling2d_5/MaxPool:output:0*
T0*0
_output_shapes
:?????????D\?2
model_1/dropout_8/Identity?
'model_1/conv2d_22/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_22_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02)
'model_1/conv2d_22/Conv2D/ReadVariableOp?
model_1/conv2d_22/Conv2D/CastCast/model_1/conv2d_22/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*(
_output_shapes
:??2
model_1/conv2d_22/Conv2D/Cast?
model_1/conv2d_22/Conv2DConv2D#model_1/dropout_8/Identity:output:0!model_1/conv2d_22/Conv2D/Cast:y:0*
T0*0
_output_shapes
:?????????D\?*
paddingSAME*
strides
2
model_1/conv2d_22/Conv2D?
(model_1/conv2d_22/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_22_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(model_1/conv2d_22/BiasAdd/ReadVariableOp?
model_1/conv2d_22/BiasAdd/CastCast0model_1/conv2d_22/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:?2 
model_1/conv2d_22/BiasAdd/Cast?
model_1/conv2d_22/BiasAddBiasAdd!model_1/conv2d_22/Conv2D:output:0"model_1/conv2d_22/BiasAdd/Cast:y:0*
T0*0
_output_shapes
:?????????D\?2
model_1/conv2d_22/BiasAdd?
-model_1/batch_normalization_21/ReadVariableOpReadVariableOp6model_1_batch_normalization_21_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-model_1/batch_normalization_21/ReadVariableOp?
/model_1/batch_normalization_21/ReadVariableOp_1ReadVariableOp8model_1_batch_normalization_21_readvariableop_1_resource*
_output_shapes	
:?*
dtype021
/model_1/batch_normalization_21/ReadVariableOp_1?
>model_1/batch_normalization_21/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_1_batch_normalization_21_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02@
>model_1/batch_normalization_21/FusedBatchNormV3/ReadVariableOp?
@model_1/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_1_batch_normalization_21_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02B
@model_1/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1?
/model_1/batch_normalization_21/FusedBatchNormV3FusedBatchNormV3"model_1/conv2d_22/BiasAdd:output:05model_1/batch_normalization_21/ReadVariableOp:value:07model_1/batch_normalization_21/ReadVariableOp_1:value:0Fmodel_1/batch_normalization_21/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_1/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????D\?:?:?:?:?:*
epsilon%o?:*
is_training( 21
/model_1/batch_normalization_21/FusedBatchNormV3?
model_1/activation_21/ReluRelu3model_1/batch_normalization_21/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????D\?2
model_1/activation_21/Relu?
 model_1/conv2d_transpose_3/ShapeShape(model_1/activation_21/Relu:activations:0*
T0*
_output_shapes
:2"
 model_1/conv2d_transpose_3/Shape?
.model_1/conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.model_1/conv2d_transpose_3/strided_slice/stack?
0model_1/conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0model_1/conv2d_transpose_3/strided_slice/stack_1?
0model_1/conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0model_1/conv2d_transpose_3/strided_slice/stack_2?
(model_1/conv2d_transpose_3/strided_sliceStridedSlice)model_1/conv2d_transpose_3/Shape:output:07model_1/conv2d_transpose_3/strided_slice/stack:output:09model_1/conv2d_transpose_3/strided_slice/stack_1:output:09model_1/conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(model_1/conv2d_transpose_3/strided_slice?
"model_1/conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2$
"model_1/conv2d_transpose_3/stack/1?
"model_1/conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2$
"model_1/conv2d_transpose_3/stack/2?
"model_1/conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2$
"model_1/conv2d_transpose_3/stack/3?
 model_1/conv2d_transpose_3/stackPack1model_1/conv2d_transpose_3/strided_slice:output:0+model_1/conv2d_transpose_3/stack/1:output:0+model_1/conv2d_transpose_3/stack/2:output:0+model_1/conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2"
 model_1/conv2d_transpose_3/stack?
0model_1/conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0model_1/conv2d_transpose_3/strided_slice_1/stack?
2model_1/conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2model_1/conv2d_transpose_3/strided_slice_1/stack_1?
2model_1/conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2model_1/conv2d_transpose_3/strided_slice_1/stack_2?
*model_1/conv2d_transpose_3/strided_slice_1StridedSlice)model_1/conv2d_transpose_3/stack:output:09model_1/conv2d_transpose_3/strided_slice_1/stack:output:0;model_1/conv2d_transpose_3/strided_slice_1/stack_1:output:0;model_1/conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*model_1/conv2d_transpose_3/strided_slice_1?
:model_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOpCmodel_1_conv2d_transpose_3_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype02<
:model_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOp?
0model_1/conv2d_transpose_3/conv2d_transpose/CastCastBmodel_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0*

DstT0*

SrcT0*(
_output_shapes
:??22
0model_1/conv2d_transpose_3/conv2d_transpose/Cast?
+model_1/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput)model_1/conv2d_transpose_3/stack:output:04model_1/conv2d_transpose_3/conv2d_transpose/Cast:y:0(model_1/activation_21/Relu:activations:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2-
+model_1/conv2d_transpose_3/conv2d_transpose?
1model_1/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp:model_1_conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype023
1model_1/conv2d_transpose_3/BiasAdd/ReadVariableOp?
'model_1/conv2d_transpose_3/BiasAdd/CastCast9model_1/conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:?2)
'model_1/conv2d_transpose_3/BiasAdd/Cast?
"model_1/conv2d_transpose_3/BiasAddBiasAdd4model_1/conv2d_transpose_3/conv2d_transpose:output:0+model_1/conv2d_transpose_3/BiasAdd/Cast:y:0*
T0*2
_output_shapes 
:????????????2$
"model_1/conv2d_transpose_3/BiasAdd?
!model_1/concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_1/concatenate_3/concat/axis?
model_1/concatenate_3/concatConcatV2+model_1/conv2d_transpose_3/BiasAdd:output:0(model_1/activation_19/Relu:activations:0*model_1/concatenate_3/concat/axis:output:0*
N*
T0*2
_output_shapes 
:????????????2
model_1/concatenate_3/concat?
model_1/dropout_9/IdentityIdentity%model_1/concatenate_3/concat:output:0*
T0*2
_output_shapes 
:????????????2
model_1/dropout_9/Identity?
'model_1/conv2d_24/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_24_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02)
'model_1/conv2d_24/Conv2D/ReadVariableOp?
model_1/conv2d_24/Conv2D/CastCast/model_1/conv2d_24/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*(
_output_shapes
:??2
model_1/conv2d_24/Conv2D/Cast?
model_1/conv2d_24/Conv2DConv2D#model_1/dropout_9/Identity:output:0!model_1/conv2d_24/Conv2D/Cast:y:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
model_1/conv2d_24/Conv2D?
(model_1/conv2d_24/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_24_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(model_1/conv2d_24/BiasAdd/ReadVariableOp?
model_1/conv2d_24/BiasAdd/CastCast0model_1/conv2d_24/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:?2 
model_1/conv2d_24/BiasAdd/Cast?
model_1/conv2d_24/BiasAddBiasAdd!model_1/conv2d_24/Conv2D:output:0"model_1/conv2d_24/BiasAdd/Cast:y:0*
T0*2
_output_shapes 
:????????????2
model_1/conv2d_24/BiasAdd?
-model_1/batch_normalization_23/ReadVariableOpReadVariableOp6model_1_batch_normalization_23_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-model_1/batch_normalization_23/ReadVariableOp?
/model_1/batch_normalization_23/ReadVariableOp_1ReadVariableOp8model_1_batch_normalization_23_readvariableop_1_resource*
_output_shapes	
:?*
dtype021
/model_1/batch_normalization_23/ReadVariableOp_1?
>model_1/batch_normalization_23/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_1_batch_normalization_23_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02@
>model_1/batch_normalization_23/FusedBatchNormV3/ReadVariableOp?
@model_1/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_1_batch_normalization_23_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02B
@model_1/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1?
/model_1/batch_normalization_23/FusedBatchNormV3FusedBatchNormV3"model_1/conv2d_24/BiasAdd:output:05model_1/batch_normalization_23/ReadVariableOp:value:07model_1/batch_normalization_23/ReadVariableOp_1:value:0Fmodel_1/batch_normalization_23/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_1/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:????????????:?:?:?:?:*
epsilon%o?:*
is_training( 21
/model_1/batch_normalization_23/FusedBatchNormV3?
model_1/activation_23/ReluRelu3model_1/batch_normalization_23/FusedBatchNormV3:y:0*
T0*2
_output_shapes 
:????????????2
model_1/activation_23/Relu?
 model_1/conv2d_transpose_4/ShapeShape(model_1/activation_23/Relu:activations:0*
T0*
_output_shapes
:2"
 model_1/conv2d_transpose_4/Shape?
.model_1/conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.model_1/conv2d_transpose_4/strided_slice/stack?
0model_1/conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0model_1/conv2d_transpose_4/strided_slice/stack_1?
0model_1/conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0model_1/conv2d_transpose_4/strided_slice/stack_2?
(model_1/conv2d_transpose_4/strided_sliceStridedSlice)model_1/conv2d_transpose_4/Shape:output:07model_1/conv2d_transpose_4/strided_slice/stack:output:09model_1/conv2d_transpose_4/strided_slice/stack_1:output:09model_1/conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(model_1/conv2d_transpose_4/strided_slice?
"model_1/conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2$
"model_1/conv2d_transpose_4/stack/1?
"model_1/conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2$
"model_1/conv2d_transpose_4/stack/2?
"model_1/conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2$
"model_1/conv2d_transpose_4/stack/3?
 model_1/conv2d_transpose_4/stackPack1model_1/conv2d_transpose_4/strided_slice:output:0+model_1/conv2d_transpose_4/stack/1:output:0+model_1/conv2d_transpose_4/stack/2:output:0+model_1/conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:2"
 model_1/conv2d_transpose_4/stack?
0model_1/conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0model_1/conv2d_transpose_4/strided_slice_1/stack?
2model_1/conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2model_1/conv2d_transpose_4/strided_slice_1/stack_1?
2model_1/conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2model_1/conv2d_transpose_4/strided_slice_1/stack_2?
*model_1/conv2d_transpose_4/strided_slice_1StridedSlice)model_1/conv2d_transpose_4/stack:output:09model_1/conv2d_transpose_4/strided_slice_1/stack:output:0;model_1/conv2d_transpose_4/strided_slice_1/stack_1:output:0;model_1/conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*model_1/conv2d_transpose_4/strided_slice_1?
:model_1/conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOpCmodel_1_conv2d_transpose_4_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype02<
:model_1/conv2d_transpose_4/conv2d_transpose/ReadVariableOp?
0model_1/conv2d_transpose_4/conv2d_transpose/CastCastBmodel_1/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0*

DstT0*

SrcT0*'
_output_shapes
:@?22
0model_1/conv2d_transpose_4/conv2d_transpose/Cast?
+model_1/conv2d_transpose_4/conv2d_transposeConv2DBackpropInput)model_1/conv2d_transpose_4/stack:output:04model_1/conv2d_transpose_4/conv2d_transpose/Cast:y:0(model_1/activation_23/Relu:activations:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2-
+model_1/conv2d_transpose_4/conv2d_transpose?
1model_1/conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp:model_1_conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype023
1model_1/conv2d_transpose_4/BiasAdd/ReadVariableOp?
'model_1/conv2d_transpose_4/BiasAdd/CastCast9model_1/conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@2)
'model_1/conv2d_transpose_4/BiasAdd/Cast?
"model_1/conv2d_transpose_4/BiasAddBiasAdd4model_1/conv2d_transpose_4/conv2d_transpose:output:0+model_1/conv2d_transpose_4/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:???????????@2$
"model_1/conv2d_transpose_4/BiasAdd?
!model_1/concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_1/concatenate_4/concat/axis?
model_1/concatenate_4/concatConcatV2+model_1/conv2d_transpose_4/BiasAdd:output:0(model_1/activation_17/Relu:activations:0*model_1/concatenate_4/concat/axis:output:0*
N*
T0*2
_output_shapes 
:????????????2
model_1/concatenate_4/concat?
model_1/dropout_10/IdentityIdentity%model_1/concatenate_4/concat:output:0*
T0*2
_output_shapes 
:????????????2
model_1/dropout_10/Identity?
'model_1/conv2d_26/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_26_conv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02)
'model_1/conv2d_26/Conv2D/ReadVariableOp?
model_1/conv2d_26/Conv2D/CastCast/model_1/conv2d_26/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*'
_output_shapes
:?@2
model_1/conv2d_26/Conv2D/Cast?
model_1/conv2d_26/Conv2DConv2D$model_1/dropout_10/Identity:output:0!model_1/conv2d_26/Conv2D/Cast:y:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
model_1/conv2d_26/Conv2D?
(model_1/conv2d_26/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_1/conv2d_26/BiasAdd/ReadVariableOp?
model_1/conv2d_26/BiasAdd/CastCast0model_1/conv2d_26/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@2 
model_1/conv2d_26/BiasAdd/Cast?
model_1/conv2d_26/BiasAddBiasAdd!model_1/conv2d_26/Conv2D:output:0"model_1/conv2d_26/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:???????????@2
model_1/conv2d_26/BiasAdd?
-model_1/batch_normalization_25/ReadVariableOpReadVariableOp6model_1_batch_normalization_25_readvariableop_resource*
_output_shapes
:@*
dtype02/
-model_1/batch_normalization_25/ReadVariableOp?
/model_1/batch_normalization_25/ReadVariableOp_1ReadVariableOp8model_1_batch_normalization_25_readvariableop_1_resource*
_output_shapes
:@*
dtype021
/model_1/batch_normalization_25/ReadVariableOp_1?
>model_1/batch_normalization_25/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_1_batch_normalization_25_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02@
>model_1/batch_normalization_25/FusedBatchNormV3/ReadVariableOp?
@model_1/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_1_batch_normalization_25_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02B
@model_1/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1?
/model_1/batch_normalization_25/FusedBatchNormV3FusedBatchNormV3"model_1/conv2d_26/BiasAdd:output:05model_1/batch_normalization_25/ReadVariableOp:value:07model_1/batch_normalization_25/ReadVariableOp_1:value:0Fmodel_1/batch_normalization_25/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_1/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
is_training( 21
/model_1/batch_normalization_25/FusedBatchNormV3?
model_1/activation_25/ReluRelu3model_1/batch_normalization_25/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????@2
model_1/activation_25/Relu?
 model_1/conv2d_transpose_5/ShapeShape(model_1/activation_25/Relu:activations:0*
T0*
_output_shapes
:2"
 model_1/conv2d_transpose_5/Shape?
.model_1/conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.model_1/conv2d_transpose_5/strided_slice/stack?
0model_1/conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0model_1/conv2d_transpose_5/strided_slice/stack_1?
0model_1/conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0model_1/conv2d_transpose_5/strided_slice/stack_2?
(model_1/conv2d_transpose_5/strided_sliceStridedSlice)model_1/conv2d_transpose_5/Shape:output:07model_1/conv2d_transpose_5/strided_slice/stack:output:09model_1/conv2d_transpose_5/strided_slice/stack_1:output:09model_1/conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(model_1/conv2d_transpose_5/strided_slice?
"model_1/conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2$
"model_1/conv2d_transpose_5/stack/1?
"model_1/conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2$
"model_1/conv2d_transpose_5/stack/2?
"model_1/conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2$
"model_1/conv2d_transpose_5/stack/3?
 model_1/conv2d_transpose_5/stackPack1model_1/conv2d_transpose_5/strided_slice:output:0+model_1/conv2d_transpose_5/stack/1:output:0+model_1/conv2d_transpose_5/stack/2:output:0+model_1/conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:2"
 model_1/conv2d_transpose_5/stack?
0model_1/conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0model_1/conv2d_transpose_5/strided_slice_1/stack?
2model_1/conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2model_1/conv2d_transpose_5/strided_slice_1/stack_1?
2model_1/conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2model_1/conv2d_transpose_5/strided_slice_1/stack_2?
*model_1/conv2d_transpose_5/strided_slice_1StridedSlice)model_1/conv2d_transpose_5/stack:output:09model_1/conv2d_transpose_5/strided_slice_1/stack:output:0;model_1/conv2d_transpose_5/strided_slice_1/stack_1:output:0;model_1/conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*model_1/conv2d_transpose_5/strided_slice_1?
:model_1/conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOpCmodel_1_conv2d_transpose_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02<
:model_1/conv2d_transpose_5/conv2d_transpose/ReadVariableOp?
0model_1/conv2d_transpose_5/conv2d_transpose/CastCastBmodel_1/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: @22
0model_1/conv2d_transpose_5/conv2d_transpose/Cast?
+model_1/conv2d_transpose_5/conv2d_transposeConv2DBackpropInput)model_1/conv2d_transpose_5/stack:output:04model_1/conv2d_transpose_5/conv2d_transpose/Cast:y:0(model_1/activation_25/Relu:activations:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2-
+model_1/conv2d_transpose_5/conv2d_transpose?
1model_1/conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp:model_1_conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype023
1model_1/conv2d_transpose_5/BiasAdd/ReadVariableOp?
'model_1/conv2d_transpose_5/BiasAdd/CastCast9model_1/conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: 2)
'model_1/conv2d_transpose_5/BiasAdd/Cast?
"model_1/conv2d_transpose_5/BiasAddBiasAdd4model_1/conv2d_transpose_5/conv2d_transpose:output:0+model_1/conv2d_transpose_5/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:??????????? 2$
"model_1/conv2d_transpose_5/BiasAdd?
!model_1/concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_1/concatenate_5/concat/axis?
model_1/concatenate_5/concatConcatV2+model_1/conv2d_transpose_5/BiasAdd:output:0(model_1/activation_15/Relu:activations:0*model_1/concatenate_5/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????@2
model_1/concatenate_5/concat?
model_1/dropout_11/IdentityIdentity%model_1/concatenate_5/concat:output:0*
T0*1
_output_shapes
:???????????@2
model_1/dropout_11/Identity?
'model_1/conv2d_28/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02)
'model_1/conv2d_28/Conv2D/ReadVariableOp?
model_1/conv2d_28/Conv2D/CastCast/model_1/conv2d_28/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:@ 2
model_1/conv2d_28/Conv2D/Cast?
model_1/conv2d_28/Conv2DConv2D$model_1/dropout_11/Identity:output:0!model_1/conv2d_28/Conv2D/Cast:y:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
model_1/conv2d_28/Conv2D?
(model_1/conv2d_28/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_28_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model_1/conv2d_28/BiasAdd/ReadVariableOp?
model_1/conv2d_28/BiasAdd/CastCast0model_1/conv2d_28/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: 2 
model_1/conv2d_28/BiasAdd/Cast?
model_1/conv2d_28/BiasAddBiasAdd!model_1/conv2d_28/Conv2D:output:0"model_1/conv2d_28/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:??????????? 2
model_1/conv2d_28/BiasAdd?
-model_1/batch_normalization_27/ReadVariableOpReadVariableOp6model_1_batch_normalization_27_readvariableop_resource*
_output_shapes
: *
dtype02/
-model_1/batch_normalization_27/ReadVariableOp?
/model_1/batch_normalization_27/ReadVariableOp_1ReadVariableOp8model_1_batch_normalization_27_readvariableop_1_resource*
_output_shapes
: *
dtype021
/model_1/batch_normalization_27/ReadVariableOp_1?
>model_1/batch_normalization_27/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_1_batch_normalization_27_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02@
>model_1/batch_normalization_27/FusedBatchNormV3/ReadVariableOp?
@model_1/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_1_batch_normalization_27_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02B
@model_1/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1?
/model_1/batch_normalization_27/FusedBatchNormV3FusedBatchNormV3"model_1/conv2d_28/BiasAdd:output:05model_1/batch_normalization_27/ReadVariableOp:value:07model_1/batch_normalization_27/ReadVariableOp_1:value:0Fmodel_1/batch_normalization_27/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_1/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 21
/model_1/batch_normalization_27/FusedBatchNormV3?
model_1/activation_27/ReluRelu3model_1/batch_normalization_27/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? 2
model_1/activation_27/Relu?
'model_1/conv2d_29/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_29_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'model_1/conv2d_29/Conv2D/ReadVariableOp?
model_1/conv2d_29/Conv2D/CastCast/model_1/conv2d_29/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: 2
model_1/conv2d_29/Conv2D/Cast?
model_1/conv2d_29/Conv2DConv2D(model_1/activation_27/Relu:activations:0!model_1/conv2d_29/Conv2D/Cast:y:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
model_1/conv2d_29/Conv2D?
(model_1/conv2d_29/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_1/conv2d_29/BiasAdd/ReadVariableOp?
model_1/conv2d_29/BiasAdd/CastCast0model_1/conv2d_29/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:2 
model_1/conv2d_29/BiasAdd/Cast?
model_1/conv2d_29/BiasAddBiasAdd!model_1/conv2d_29/Conv2D:output:0"model_1/conv2d_29/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:???????????2
model_1/conv2d_29/BiasAdd?
model_1/conv2d_29/SigmoidSigmoid"model_1/conv2d_29/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
model_1/conv2d_29/Sigmoid?
IdentityIdentitymodel_1/conv2d_29/Sigmoid:y:0?^model_1/batch_normalization_15/FusedBatchNormV3/ReadVariableOpA^model_1/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1.^model_1/batch_normalization_15/ReadVariableOp0^model_1/batch_normalization_15/ReadVariableOp_1?^model_1/batch_normalization_17/FusedBatchNormV3/ReadVariableOpA^model_1/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1.^model_1/batch_normalization_17/ReadVariableOp0^model_1/batch_normalization_17/ReadVariableOp_1?^model_1/batch_normalization_19/FusedBatchNormV3/ReadVariableOpA^model_1/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1.^model_1/batch_normalization_19/ReadVariableOp0^model_1/batch_normalization_19/ReadVariableOp_1?^model_1/batch_normalization_21/FusedBatchNormV3/ReadVariableOpA^model_1/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1.^model_1/batch_normalization_21/ReadVariableOp0^model_1/batch_normalization_21/ReadVariableOp_1?^model_1/batch_normalization_23/FusedBatchNormV3/ReadVariableOpA^model_1/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1.^model_1/batch_normalization_23/ReadVariableOp0^model_1/batch_normalization_23/ReadVariableOp_1?^model_1/batch_normalization_25/FusedBatchNormV3/ReadVariableOpA^model_1/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1.^model_1/batch_normalization_25/ReadVariableOp0^model_1/batch_normalization_25/ReadVariableOp_1?^model_1/batch_normalization_27/FusedBatchNormV3/ReadVariableOpA^model_1/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1.^model_1/batch_normalization_27/ReadVariableOp0^model_1/batch_normalization_27/ReadVariableOp_1)^model_1/conv2d_16/BiasAdd/ReadVariableOp(^model_1/conv2d_16/Conv2D/ReadVariableOp)^model_1/conv2d_18/BiasAdd/ReadVariableOp(^model_1/conv2d_18/Conv2D/ReadVariableOp)^model_1/conv2d_20/BiasAdd/ReadVariableOp(^model_1/conv2d_20/Conv2D/ReadVariableOp)^model_1/conv2d_22/BiasAdd/ReadVariableOp(^model_1/conv2d_22/Conv2D/ReadVariableOp)^model_1/conv2d_24/BiasAdd/ReadVariableOp(^model_1/conv2d_24/Conv2D/ReadVariableOp)^model_1/conv2d_26/BiasAdd/ReadVariableOp(^model_1/conv2d_26/Conv2D/ReadVariableOp)^model_1/conv2d_28/BiasAdd/ReadVariableOp(^model_1/conv2d_28/Conv2D/ReadVariableOp)^model_1/conv2d_29/BiasAdd/ReadVariableOp(^model_1/conv2d_29/Conv2D/ReadVariableOp2^model_1/conv2d_transpose_3/BiasAdd/ReadVariableOp;^model_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOp2^model_1/conv2d_transpose_4/BiasAdd/ReadVariableOp;^model_1/conv2d_transpose_4/conv2d_transpose/ReadVariableOp2^model_1/conv2d_transpose_5/BiasAdd/ReadVariableOp;^model_1/conv2d_transpose_5/conv2d_transpose/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::::::::::2?
>model_1/batch_normalization_15/FusedBatchNormV3/ReadVariableOp>model_1/batch_normalization_15/FusedBatchNormV3/ReadVariableOp2?
@model_1/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1@model_1/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_12^
-model_1/batch_normalization_15/ReadVariableOp-model_1/batch_normalization_15/ReadVariableOp2b
/model_1/batch_normalization_15/ReadVariableOp_1/model_1/batch_normalization_15/ReadVariableOp_12?
>model_1/batch_normalization_17/FusedBatchNormV3/ReadVariableOp>model_1/batch_normalization_17/FusedBatchNormV3/ReadVariableOp2?
@model_1/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1@model_1/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12^
-model_1/batch_normalization_17/ReadVariableOp-model_1/batch_normalization_17/ReadVariableOp2b
/model_1/batch_normalization_17/ReadVariableOp_1/model_1/batch_normalization_17/ReadVariableOp_12?
>model_1/batch_normalization_19/FusedBatchNormV3/ReadVariableOp>model_1/batch_normalization_19/FusedBatchNormV3/ReadVariableOp2?
@model_1/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1@model_1/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_12^
-model_1/batch_normalization_19/ReadVariableOp-model_1/batch_normalization_19/ReadVariableOp2b
/model_1/batch_normalization_19/ReadVariableOp_1/model_1/batch_normalization_19/ReadVariableOp_12?
>model_1/batch_normalization_21/FusedBatchNormV3/ReadVariableOp>model_1/batch_normalization_21/FusedBatchNormV3/ReadVariableOp2?
@model_1/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1@model_1/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_12^
-model_1/batch_normalization_21/ReadVariableOp-model_1/batch_normalization_21/ReadVariableOp2b
/model_1/batch_normalization_21/ReadVariableOp_1/model_1/batch_normalization_21/ReadVariableOp_12?
>model_1/batch_normalization_23/FusedBatchNormV3/ReadVariableOp>model_1/batch_normalization_23/FusedBatchNormV3/ReadVariableOp2?
@model_1/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1@model_1/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_12^
-model_1/batch_normalization_23/ReadVariableOp-model_1/batch_normalization_23/ReadVariableOp2b
/model_1/batch_normalization_23/ReadVariableOp_1/model_1/batch_normalization_23/ReadVariableOp_12?
>model_1/batch_normalization_25/FusedBatchNormV3/ReadVariableOp>model_1/batch_normalization_25/FusedBatchNormV3/ReadVariableOp2?
@model_1/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1@model_1/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_12^
-model_1/batch_normalization_25/ReadVariableOp-model_1/batch_normalization_25/ReadVariableOp2b
/model_1/batch_normalization_25/ReadVariableOp_1/model_1/batch_normalization_25/ReadVariableOp_12?
>model_1/batch_normalization_27/FusedBatchNormV3/ReadVariableOp>model_1/batch_normalization_27/FusedBatchNormV3/ReadVariableOp2?
@model_1/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1@model_1/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_12^
-model_1/batch_normalization_27/ReadVariableOp-model_1/batch_normalization_27/ReadVariableOp2b
/model_1/batch_normalization_27/ReadVariableOp_1/model_1/batch_normalization_27/ReadVariableOp_12T
(model_1/conv2d_16/BiasAdd/ReadVariableOp(model_1/conv2d_16/BiasAdd/ReadVariableOp2R
'model_1/conv2d_16/Conv2D/ReadVariableOp'model_1/conv2d_16/Conv2D/ReadVariableOp2T
(model_1/conv2d_18/BiasAdd/ReadVariableOp(model_1/conv2d_18/BiasAdd/ReadVariableOp2R
'model_1/conv2d_18/Conv2D/ReadVariableOp'model_1/conv2d_18/Conv2D/ReadVariableOp2T
(model_1/conv2d_20/BiasAdd/ReadVariableOp(model_1/conv2d_20/BiasAdd/ReadVariableOp2R
'model_1/conv2d_20/Conv2D/ReadVariableOp'model_1/conv2d_20/Conv2D/ReadVariableOp2T
(model_1/conv2d_22/BiasAdd/ReadVariableOp(model_1/conv2d_22/BiasAdd/ReadVariableOp2R
'model_1/conv2d_22/Conv2D/ReadVariableOp'model_1/conv2d_22/Conv2D/ReadVariableOp2T
(model_1/conv2d_24/BiasAdd/ReadVariableOp(model_1/conv2d_24/BiasAdd/ReadVariableOp2R
'model_1/conv2d_24/Conv2D/ReadVariableOp'model_1/conv2d_24/Conv2D/ReadVariableOp2T
(model_1/conv2d_26/BiasAdd/ReadVariableOp(model_1/conv2d_26/BiasAdd/ReadVariableOp2R
'model_1/conv2d_26/Conv2D/ReadVariableOp'model_1/conv2d_26/Conv2D/ReadVariableOp2T
(model_1/conv2d_28/BiasAdd/ReadVariableOp(model_1/conv2d_28/BiasAdd/ReadVariableOp2R
'model_1/conv2d_28/Conv2D/ReadVariableOp'model_1/conv2d_28/Conv2D/ReadVariableOp2T
(model_1/conv2d_29/BiasAdd/ReadVariableOp(model_1/conv2d_29/BiasAdd/ReadVariableOp2R
'model_1/conv2d_29/Conv2D/ReadVariableOp'model_1/conv2d_29/Conv2D/ReadVariableOp2f
1model_1/conv2d_transpose_3/BiasAdd/ReadVariableOp1model_1/conv2d_transpose_3/BiasAdd/ReadVariableOp2x
:model_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:model_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOp2f
1model_1/conv2d_transpose_4/BiasAdd/ReadVariableOp1model_1/conv2d_transpose_4/BiasAdd/ReadVariableOp2x
:model_1/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:model_1/conv2d_transpose_4/conv2d_transpose/ReadVariableOp2f
1model_1/conv2d_transpose_5/BiasAdd/ReadVariableOp1model_1/conv2d_transpose_5/BiasAdd/ReadVariableOp2x
:model_1/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:model_1/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_2
?
d
H__inference_activation_25_layer_call_and_return_conditional_losses_39308

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:???????????@2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????@:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
c
D__inference_dropout_7_layer_call_and_return_conditional_losses_41392

inputs
identity?a
dropout/ConstConst*
_output_shapes
: *
dtype0*
value
B j?x2
dropout/Const}
dropout/MulMulinputsdropout/Const:output:0*
T0*1
_output_shapes
:???????????@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*1
_output_shapes
:???????????@*
dtype02&
$dropout/random_uniform/RandomUniforms
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value
B j?\2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:???????????@2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:???????????@2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*1
_output_shapes
:???????????@2
dropout/Mul_1o
IdentityIdentitydropout/Mul_1:z:0*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????@:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
D__inference_conv2d_28_layer_call_and_return_conditional_losses_42202

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02
Conv2D/ReadVariableOp?
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:@ 2
Conv2D/Cast?
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpx
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: 2
BiasAdd/Cast|
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*1
_output_shapes
:??????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_27_layer_call_fn_42262

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_27_layer_call_and_return_conditional_losses_394142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:??????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_25_layer_call_and_return_conditional_losses_42096

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?%
?
M__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_38318

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transpose/CastCast'conv2d_transpose/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: @2
conv2d_transpose/Cast?
conv2d_transposeConv2DBackpropInputstack:output:0conv2d_transpose/Cast:y:0inputs*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpx
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: 2
BiasAdd/Cast?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/Cast:y:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
D__inference_conv2d_16_layer_call_and_return_conditional_losses_38449

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: 2
Conv2D/Cast?
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpx
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: 2
BiasAdd/Cast|
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*1
_output_shapes
:??????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
2__inference_conv2d_transpose_5_layer_call_fn_38328

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *V
fQRO
M__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_383182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
d
H__inference_activation_27_layer_call_and_return_conditional_losses_39473

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:??????????? 2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
D__inference_conv2d_18_layer_call_and_return_conditional_losses_38594

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: @2
Conv2D/Cast?
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpx
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@2
BiasAdd/Cast|
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*1
_output_shapes
:???????????@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:??????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
d
H__inference_activation_15_layer_call_and_return_conditional_losses_38543

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:??????????? 2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
I
-__inference_activation_25_layer_call_fn_42150

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *Q
fLRJ
H__inference_activation_25_layer_call_and_return_conditional_losses_393082
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????@:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
d
E__inference_dropout_10_layer_call_and_return_conditional_losses_39184

inputs
identity?a
dropout/ConstConst*
_output_shapes
: *
dtype0*
value
B j?x2
dropout/Const~
dropout/MulMulinputsdropout/Const:output:0*
T0*2
_output_shapes 
:????????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*2
_output_shapes 
:????????????*
dtype02&
$dropout/random_uniform/RandomUniforms
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value
B j?\2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*2
_output_shapes 
:????????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*2
_output_shapes 
:????????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*2
_output_shapes 
:????????????2
dropout/Mul_1p
IdentityIdentitydropout/Mul_1:z:0*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*1
_input_shapes 
:????????????:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_21_layer_call_fn_41678

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_21_layer_call_and_return_conditional_losses_379712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_41512

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
??
?Q
!__inference__traced_restore_43202
file_prefix%
!assignvariableop_conv2d_16_kernel%
!assignvariableop_1_conv2d_16_bias3
/assignvariableop_2_batch_normalization_15_gamma2
.assignvariableop_3_batch_normalization_15_beta9
5assignvariableop_4_batch_normalization_15_moving_mean=
9assignvariableop_5_batch_normalization_15_moving_variance'
#assignvariableop_6_conv2d_18_kernel%
!assignvariableop_7_conv2d_18_bias3
/assignvariableop_8_batch_normalization_17_gamma2
.assignvariableop_9_batch_normalization_17_beta:
6assignvariableop_10_batch_normalization_17_moving_mean>
:assignvariableop_11_batch_normalization_17_moving_variance(
$assignvariableop_12_conv2d_20_kernel&
"assignvariableop_13_conv2d_20_bias4
0assignvariableop_14_batch_normalization_19_gamma3
/assignvariableop_15_batch_normalization_19_beta:
6assignvariableop_16_batch_normalization_19_moving_mean>
:assignvariableop_17_batch_normalization_19_moving_variance(
$assignvariableop_18_conv2d_22_kernel&
"assignvariableop_19_conv2d_22_bias4
0assignvariableop_20_batch_normalization_21_gamma3
/assignvariableop_21_batch_normalization_21_beta:
6assignvariableop_22_batch_normalization_21_moving_mean>
:assignvariableop_23_batch_normalization_21_moving_variance1
-assignvariableop_24_conv2d_transpose_3_kernel/
+assignvariableop_25_conv2d_transpose_3_bias(
$assignvariableop_26_conv2d_24_kernel&
"assignvariableop_27_conv2d_24_bias4
0assignvariableop_28_batch_normalization_23_gamma3
/assignvariableop_29_batch_normalization_23_beta:
6assignvariableop_30_batch_normalization_23_moving_mean>
:assignvariableop_31_batch_normalization_23_moving_variance1
-assignvariableop_32_conv2d_transpose_4_kernel/
+assignvariableop_33_conv2d_transpose_4_bias(
$assignvariableop_34_conv2d_26_kernel&
"assignvariableop_35_conv2d_26_bias4
0assignvariableop_36_batch_normalization_25_gamma3
/assignvariableop_37_batch_normalization_25_beta:
6assignvariableop_38_batch_normalization_25_moving_mean>
:assignvariableop_39_batch_normalization_25_moving_variance1
-assignvariableop_40_conv2d_transpose_5_kernel/
+assignvariableop_41_conv2d_transpose_5_bias(
$assignvariableop_42_conv2d_28_kernel&
"assignvariableop_43_conv2d_28_bias4
0assignvariableop_44_batch_normalization_27_gamma3
/assignvariableop_45_batch_normalization_27_beta:
6assignvariableop_46_batch_normalization_27_moving_mean>
:assignvariableop_47_batch_normalization_27_moving_variance(
$assignvariableop_48_conv2d_29_kernel&
"assignvariableop_49_conv2d_29_bias(
$assignvariableop_50_cond_1_adam_iter*
&assignvariableop_51_cond_1_adam_beta_1*
&assignvariableop_52_cond_1_adam_beta_2)
%assignvariableop_53_cond_1_adam_decay1
-assignvariableop_54_cond_1_adam_learning_rate*
&assignvariableop_55_current_loss_scale"
assignvariableop_56_good_steps
assignvariableop_57_total
assignvariableop_58_count
assignvariableop_59_total_1
assignvariableop_60_count_16
2assignvariableop_61_cond_1_adam_conv2d_16_kernel_m4
0assignvariableop_62_cond_1_adam_conv2d_16_bias_mB
>assignvariableop_63_cond_1_adam_batch_normalization_15_gamma_mA
=assignvariableop_64_cond_1_adam_batch_normalization_15_beta_m6
2assignvariableop_65_cond_1_adam_conv2d_18_kernel_m4
0assignvariableop_66_cond_1_adam_conv2d_18_bias_mB
>assignvariableop_67_cond_1_adam_batch_normalization_17_gamma_mA
=assignvariableop_68_cond_1_adam_batch_normalization_17_beta_m6
2assignvariableop_69_cond_1_adam_conv2d_20_kernel_m4
0assignvariableop_70_cond_1_adam_conv2d_20_bias_mB
>assignvariableop_71_cond_1_adam_batch_normalization_19_gamma_mA
=assignvariableop_72_cond_1_adam_batch_normalization_19_beta_m6
2assignvariableop_73_cond_1_adam_conv2d_22_kernel_m4
0assignvariableop_74_cond_1_adam_conv2d_22_bias_mB
>assignvariableop_75_cond_1_adam_batch_normalization_21_gamma_mA
=assignvariableop_76_cond_1_adam_batch_normalization_21_beta_m?
;assignvariableop_77_cond_1_adam_conv2d_transpose_3_kernel_m=
9assignvariableop_78_cond_1_adam_conv2d_transpose_3_bias_m6
2assignvariableop_79_cond_1_adam_conv2d_24_kernel_m4
0assignvariableop_80_cond_1_adam_conv2d_24_bias_mB
>assignvariableop_81_cond_1_adam_batch_normalization_23_gamma_mA
=assignvariableop_82_cond_1_adam_batch_normalization_23_beta_m?
;assignvariableop_83_cond_1_adam_conv2d_transpose_4_kernel_m=
9assignvariableop_84_cond_1_adam_conv2d_transpose_4_bias_m6
2assignvariableop_85_cond_1_adam_conv2d_26_kernel_m4
0assignvariableop_86_cond_1_adam_conv2d_26_bias_mB
>assignvariableop_87_cond_1_adam_batch_normalization_25_gamma_mA
=assignvariableop_88_cond_1_adam_batch_normalization_25_beta_m?
;assignvariableop_89_cond_1_adam_conv2d_transpose_5_kernel_m=
9assignvariableop_90_cond_1_adam_conv2d_transpose_5_bias_m6
2assignvariableop_91_cond_1_adam_conv2d_28_kernel_m4
0assignvariableop_92_cond_1_adam_conv2d_28_bias_mB
>assignvariableop_93_cond_1_adam_batch_normalization_27_gamma_mA
=assignvariableop_94_cond_1_adam_batch_normalization_27_beta_m6
2assignvariableop_95_cond_1_adam_conv2d_29_kernel_m4
0assignvariableop_96_cond_1_adam_conv2d_29_bias_m6
2assignvariableop_97_cond_1_adam_conv2d_16_kernel_v4
0assignvariableop_98_cond_1_adam_conv2d_16_bias_vB
>assignvariableop_99_cond_1_adam_batch_normalization_15_gamma_vB
>assignvariableop_100_cond_1_adam_batch_normalization_15_beta_v7
3assignvariableop_101_cond_1_adam_conv2d_18_kernel_v5
1assignvariableop_102_cond_1_adam_conv2d_18_bias_vC
?assignvariableop_103_cond_1_adam_batch_normalization_17_gamma_vB
>assignvariableop_104_cond_1_adam_batch_normalization_17_beta_v7
3assignvariableop_105_cond_1_adam_conv2d_20_kernel_v5
1assignvariableop_106_cond_1_adam_conv2d_20_bias_vC
?assignvariableop_107_cond_1_adam_batch_normalization_19_gamma_vB
>assignvariableop_108_cond_1_adam_batch_normalization_19_beta_v7
3assignvariableop_109_cond_1_adam_conv2d_22_kernel_v5
1assignvariableop_110_cond_1_adam_conv2d_22_bias_vC
?assignvariableop_111_cond_1_adam_batch_normalization_21_gamma_vB
>assignvariableop_112_cond_1_adam_batch_normalization_21_beta_v@
<assignvariableop_113_cond_1_adam_conv2d_transpose_3_kernel_v>
:assignvariableop_114_cond_1_adam_conv2d_transpose_3_bias_v7
3assignvariableop_115_cond_1_adam_conv2d_24_kernel_v5
1assignvariableop_116_cond_1_adam_conv2d_24_bias_vC
?assignvariableop_117_cond_1_adam_batch_normalization_23_gamma_vB
>assignvariableop_118_cond_1_adam_batch_normalization_23_beta_v@
<assignvariableop_119_cond_1_adam_conv2d_transpose_4_kernel_v>
:assignvariableop_120_cond_1_adam_conv2d_transpose_4_bias_v7
3assignvariableop_121_cond_1_adam_conv2d_26_kernel_v5
1assignvariableop_122_cond_1_adam_conv2d_26_bias_vC
?assignvariableop_123_cond_1_adam_batch_normalization_25_gamma_vB
>assignvariableop_124_cond_1_adam_batch_normalization_25_beta_v@
<assignvariableop_125_cond_1_adam_conv2d_transpose_5_kernel_v>
:assignvariableop_126_cond_1_adam_conv2d_transpose_5_bias_v7
3assignvariableop_127_cond_1_adam_conv2d_28_kernel_v5
1assignvariableop_128_cond_1_adam_conv2d_28_bias_vC
?assignvariableop_129_cond_1_adam_batch_normalization_27_gamma_vB
>assignvariableop_130_cond_1_adam_batch_normalization_27_beta_v7
3assignvariableop_131_cond_1_adam_conv2d_29_kernel_v5
1assignvariableop_132_cond_1_adam_conv2d_29_bias_v
identity_134??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_100?AssignVariableOp_101?AssignVariableOp_102?AssignVariableOp_103?AssignVariableOp_104?AssignVariableOp_105?AssignVariableOp_106?AssignVariableOp_107?AssignVariableOp_108?AssignVariableOp_109?AssignVariableOp_11?AssignVariableOp_110?AssignVariableOp_111?AssignVariableOp_112?AssignVariableOp_113?AssignVariableOp_114?AssignVariableOp_115?AssignVariableOp_116?AssignVariableOp_117?AssignVariableOp_118?AssignVariableOp_119?AssignVariableOp_12?AssignVariableOp_120?AssignVariableOp_121?AssignVariableOp_122?AssignVariableOp_123?AssignVariableOp_124?AssignVariableOp_125?AssignVariableOp_126?AssignVariableOp_127?AssignVariableOp_128?AssignVariableOp_129?AssignVariableOp_13?AssignVariableOp_130?AssignVariableOp_131?AssignVariableOp_132?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_87?AssignVariableOp_88?AssignVariableOp_89?AssignVariableOp_9?AssignVariableOp_90?AssignVariableOp_91?AssignVariableOp_92?AssignVariableOp_93?AssignVariableOp_94?AssignVariableOp_95?AssignVariableOp_96?AssignVariableOp_97?AssignVariableOp_98?AssignVariableOp_99?K
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?J
value?JB?J?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-16/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-16/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBBoptimizer/loss_scale/current_loss_scale/.ATTRIBUTES/VARIABLE_VALUEB:optimizer/loss_scale/good_steps/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*?
dtypes?
?2?		2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_16_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_16_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_15_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_15_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_15_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_15_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_18_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_18_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_17_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_17_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_17_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_17_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp$assignvariableop_12_conv2d_20_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp"assignvariableop_13_conv2d_20_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp0assignvariableop_14_batch_normalization_19_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp/assignvariableop_15_batch_normalization_19_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp6assignvariableop_16_batch_normalization_19_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp:assignvariableop_17_batch_normalization_19_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp$assignvariableop_18_conv2d_22_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp"assignvariableop_19_conv2d_22_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp0assignvariableop_20_batch_normalization_21_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp/assignvariableop_21_batch_normalization_21_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp6assignvariableop_22_batch_normalization_21_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp:assignvariableop_23_batch_normalization_21_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp-assignvariableop_24_conv2d_transpose_3_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp+assignvariableop_25_conv2d_transpose_3_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp$assignvariableop_26_conv2d_24_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp"assignvariableop_27_conv2d_24_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp0assignvariableop_28_batch_normalization_23_gammaIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp/assignvariableop_29_batch_normalization_23_betaIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp6assignvariableop_30_batch_normalization_23_moving_meanIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp:assignvariableop_31_batch_normalization_23_moving_varianceIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp-assignvariableop_32_conv2d_transpose_4_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp+assignvariableop_33_conv2d_transpose_4_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp$assignvariableop_34_conv2d_26_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp"assignvariableop_35_conv2d_26_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp0assignvariableop_36_batch_normalization_25_gammaIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp/assignvariableop_37_batch_normalization_25_betaIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp6assignvariableop_38_batch_normalization_25_moving_meanIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp:assignvariableop_39_batch_normalization_25_moving_varianceIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp-assignvariableop_40_conv2d_transpose_5_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp+assignvariableop_41_conv2d_transpose_5_biasIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp$assignvariableop_42_conv2d_28_kernelIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp"assignvariableop_43_conv2d_28_biasIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp0assignvariableop_44_batch_normalization_27_gammaIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp/assignvariableop_45_batch_normalization_27_betaIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp6assignvariableop_46_batch_normalization_27_moving_meanIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp:assignvariableop_47_batch_normalization_27_moving_varianceIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp$assignvariableop_48_conv2d_29_kernelIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp"assignvariableop_49_conv2d_29_biasIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp$assignvariableop_50_cond_1_adam_iterIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp&assignvariableop_51_cond_1_adam_beta_1Identity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp&assignvariableop_52_cond_1_adam_beta_2Identity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp%assignvariableop_53_cond_1_adam_decayIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp-assignvariableop_54_cond_1_adam_learning_rateIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp&assignvariableop_55_current_loss_scaleIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOpassignvariableop_56_good_stepsIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOpassignvariableop_57_totalIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOpassignvariableop_58_countIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOpassignvariableop_59_total_1Identity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOpassignvariableop_60_count_1Identity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp2assignvariableop_61_cond_1_adam_conv2d_16_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp0assignvariableop_62_cond_1_adam_conv2d_16_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp>assignvariableop_63_cond_1_adam_batch_normalization_15_gamma_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp=assignvariableop_64_cond_1_adam_batch_normalization_15_beta_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp2assignvariableop_65_cond_1_adam_conv2d_18_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp0assignvariableop_66_cond_1_adam_conv2d_18_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp>assignvariableop_67_cond_1_adam_batch_normalization_17_gamma_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp=assignvariableop_68_cond_1_adam_batch_normalization_17_beta_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp2assignvariableop_69_cond_1_adam_conv2d_20_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp0assignvariableop_70_cond_1_adam_conv2d_20_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp>assignvariableop_71_cond_1_adam_batch_normalization_19_gamma_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp=assignvariableop_72_cond_1_adam_batch_normalization_19_beta_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOp2assignvariableop_73_cond_1_adam_conv2d_22_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp0assignvariableop_74_cond_1_adam_conv2d_22_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOp>assignvariableop_75_cond_1_adam_batch_normalization_21_gamma_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOp=assignvariableop_76_cond_1_adam_batch_normalization_21_beta_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOp;assignvariableop_77_cond_1_adam_conv2d_transpose_3_kernel_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOp9assignvariableop_78_cond_1_adam_conv2d_transpose_3_bias_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79?
AssignVariableOp_79AssignVariableOp2assignvariableop_79_cond_1_adam_conv2d_24_kernel_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80?
AssignVariableOp_80AssignVariableOp0assignvariableop_80_cond_1_adam_conv2d_24_bias_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81?
AssignVariableOp_81AssignVariableOp>assignvariableop_81_cond_1_adam_batch_normalization_23_gamma_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82?
AssignVariableOp_82AssignVariableOp=assignvariableop_82_cond_1_adam_batch_normalization_23_beta_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83?
AssignVariableOp_83AssignVariableOp;assignvariableop_83_cond_1_adam_conv2d_transpose_4_kernel_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84?
AssignVariableOp_84AssignVariableOp9assignvariableop_84_cond_1_adam_conv2d_transpose_4_bias_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85?
AssignVariableOp_85AssignVariableOp2assignvariableop_85_cond_1_adam_conv2d_26_kernel_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86?
AssignVariableOp_86AssignVariableOp0assignvariableop_86_cond_1_adam_conv2d_26_bias_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87?
AssignVariableOp_87AssignVariableOp>assignvariableop_87_cond_1_adam_batch_normalization_25_gamma_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88?
AssignVariableOp_88AssignVariableOp=assignvariableop_88_cond_1_adam_batch_normalization_25_beta_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89?
AssignVariableOp_89AssignVariableOp;assignvariableop_89_cond_1_adam_conv2d_transpose_5_kernel_mIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90?
AssignVariableOp_90AssignVariableOp9assignvariableop_90_cond_1_adam_conv2d_transpose_5_bias_mIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91?
AssignVariableOp_91AssignVariableOp2assignvariableop_91_cond_1_adam_conv2d_28_kernel_mIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92?
AssignVariableOp_92AssignVariableOp0assignvariableop_92_cond_1_adam_conv2d_28_bias_mIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93?
AssignVariableOp_93AssignVariableOp>assignvariableop_93_cond_1_adam_batch_normalization_27_gamma_mIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94?
AssignVariableOp_94AssignVariableOp=assignvariableop_94_cond_1_adam_batch_normalization_27_beta_mIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95?
AssignVariableOp_95AssignVariableOp2assignvariableop_95_cond_1_adam_conv2d_29_kernel_mIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96?
AssignVariableOp_96AssignVariableOp0assignvariableop_96_cond_1_adam_conv2d_29_bias_mIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97?
AssignVariableOp_97AssignVariableOp2assignvariableop_97_cond_1_adam_conv2d_16_kernel_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98?
AssignVariableOp_98AssignVariableOp0assignvariableop_98_cond_1_adam_conv2d_16_bias_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_98n
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:2
Identity_99?
AssignVariableOp_99AssignVariableOp>assignvariableop_99_cond_1_adam_batch_normalization_15_gamma_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99q
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:2
Identity_100?
AssignVariableOp_100AssignVariableOp>assignvariableop_100_cond_1_adam_batch_normalization_15_beta_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_100q
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:2
Identity_101?
AssignVariableOp_101AssignVariableOp3assignvariableop_101_cond_1_adam_conv2d_18_kernel_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_101q
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:2
Identity_102?
AssignVariableOp_102AssignVariableOp1assignvariableop_102_cond_1_adam_conv2d_18_bias_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_102q
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:2
Identity_103?
AssignVariableOp_103AssignVariableOp?assignvariableop_103_cond_1_adam_batch_normalization_17_gamma_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_103q
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:2
Identity_104?
AssignVariableOp_104AssignVariableOp>assignvariableop_104_cond_1_adam_batch_normalization_17_beta_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_104q
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:2
Identity_105?
AssignVariableOp_105AssignVariableOp3assignvariableop_105_cond_1_adam_conv2d_20_kernel_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_105q
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:2
Identity_106?
AssignVariableOp_106AssignVariableOp1assignvariableop_106_cond_1_adam_conv2d_20_bias_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_106q
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:2
Identity_107?
AssignVariableOp_107AssignVariableOp?assignvariableop_107_cond_1_adam_batch_normalization_19_gamma_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_107q
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:2
Identity_108?
AssignVariableOp_108AssignVariableOp>assignvariableop_108_cond_1_adam_batch_normalization_19_beta_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_108q
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:2
Identity_109?
AssignVariableOp_109AssignVariableOp3assignvariableop_109_cond_1_adam_conv2d_22_kernel_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_109q
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:2
Identity_110?
AssignVariableOp_110AssignVariableOp1assignvariableop_110_cond_1_adam_conv2d_22_bias_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_110q
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:2
Identity_111?
AssignVariableOp_111AssignVariableOp?assignvariableop_111_cond_1_adam_batch_normalization_21_gamma_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_111q
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:2
Identity_112?
AssignVariableOp_112AssignVariableOp>assignvariableop_112_cond_1_adam_batch_normalization_21_beta_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_112q
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:2
Identity_113?
AssignVariableOp_113AssignVariableOp<assignvariableop_113_cond_1_adam_conv2d_transpose_3_kernel_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_113q
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:2
Identity_114?
AssignVariableOp_114AssignVariableOp:assignvariableop_114_cond_1_adam_conv2d_transpose_3_bias_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_114q
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:2
Identity_115?
AssignVariableOp_115AssignVariableOp3assignvariableop_115_cond_1_adam_conv2d_24_kernel_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_115q
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:2
Identity_116?
AssignVariableOp_116AssignVariableOp1assignvariableop_116_cond_1_adam_conv2d_24_bias_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_116q
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:2
Identity_117?
AssignVariableOp_117AssignVariableOp?assignvariableop_117_cond_1_adam_batch_normalization_23_gamma_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_117q
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:2
Identity_118?
AssignVariableOp_118AssignVariableOp>assignvariableop_118_cond_1_adam_batch_normalization_23_beta_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_118q
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:2
Identity_119?
AssignVariableOp_119AssignVariableOp<assignvariableop_119_cond_1_adam_conv2d_transpose_4_kernel_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119q
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:2
Identity_120?
AssignVariableOp_120AssignVariableOp:assignvariableop_120_cond_1_adam_conv2d_transpose_4_bias_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_120q
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:2
Identity_121?
AssignVariableOp_121AssignVariableOp3assignvariableop_121_cond_1_adam_conv2d_26_kernel_vIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_121q
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:2
Identity_122?
AssignVariableOp_122AssignVariableOp1assignvariableop_122_cond_1_adam_conv2d_26_bias_vIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_122q
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:2
Identity_123?
AssignVariableOp_123AssignVariableOp?assignvariableop_123_cond_1_adam_batch_normalization_25_gamma_vIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_123q
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:2
Identity_124?
AssignVariableOp_124AssignVariableOp>assignvariableop_124_cond_1_adam_batch_normalization_25_beta_vIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_124q
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:2
Identity_125?
AssignVariableOp_125AssignVariableOp<assignvariableop_125_cond_1_adam_conv2d_transpose_5_kernel_vIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_125q
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:2
Identity_126?
AssignVariableOp_126AssignVariableOp:assignvariableop_126_cond_1_adam_conv2d_transpose_5_bias_vIdentity_126:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_126q
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:2
Identity_127?
AssignVariableOp_127AssignVariableOp3assignvariableop_127_cond_1_adam_conv2d_28_kernel_vIdentity_127:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_127q
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:2
Identity_128?
AssignVariableOp_128AssignVariableOp1assignvariableop_128_cond_1_adam_conv2d_28_bias_vIdentity_128:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_128q
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:2
Identity_129?
AssignVariableOp_129AssignVariableOp?assignvariableop_129_cond_1_adam_batch_normalization_27_gamma_vIdentity_129:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_129q
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:2
Identity_130?
AssignVariableOp_130AssignVariableOp>assignvariableop_130_cond_1_adam_batch_normalization_27_beta_vIdentity_130:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_130q
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:2
Identity_131?
AssignVariableOp_131AssignVariableOp3assignvariableop_131_cond_1_adam_conv2d_29_kernel_vIdentity_131:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_131q
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0*
_output_shapes
:2
Identity_132?
AssignVariableOp_132AssignVariableOp1assignvariableop_132_cond_1_adam_conv2d_29_bias_vIdentity_132:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1329
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_133Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_133?
Identity_134IdentityIdentity_133:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*
T0*
_output_shapes
: 2
Identity_134"%
identity_134Identity_134:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182,
AssignVariableOp_119AssignVariableOp_1192*
AssignVariableOp_12AssignVariableOp_122,
AssignVariableOp_120AssignVariableOp_1202,
AssignVariableOp_121AssignVariableOp_1212,
AssignVariableOp_122AssignVariableOp_1222,
AssignVariableOp_123AssignVariableOp_1232,
AssignVariableOp_124AssignVariableOp_1242,
AssignVariableOp_125AssignVariableOp_1252,
AssignVariableOp_126AssignVariableOp_1262,
AssignVariableOp_127AssignVariableOp_1272,
AssignVariableOp_128AssignVariableOp_1282,
AssignVariableOp_129AssignVariableOp_1292*
AssignVariableOp_13AssignVariableOp_132,
AssignVariableOp_130AssignVariableOp_1302,
AssignVariableOp_131AssignVariableOp_1312,
AssignVariableOp_132AssignVariableOp_1322*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
??
?(
B__inference_model_1_layer_call_and_return_conditional_losses_40825

inputs,
(conv2d_16_conv2d_readvariableop_resource-
)conv2d_16_biasadd_readvariableop_resource2
.batch_normalization_15_readvariableop_resource4
0batch_normalization_15_readvariableop_1_resourceC
?batch_normalization_15_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_18_conv2d_readvariableop_resource-
)conv2d_18_biasadd_readvariableop_resource2
.batch_normalization_17_readvariableop_resource4
0batch_normalization_17_readvariableop_1_resourceC
?batch_normalization_17_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_20_conv2d_readvariableop_resource-
)conv2d_20_biasadd_readvariableop_resource2
.batch_normalization_19_readvariableop_resource4
0batch_normalization_19_readvariableop_1_resourceC
?batch_normalization_19_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_22_conv2d_readvariableop_resource-
)conv2d_22_biasadd_readvariableop_resource2
.batch_normalization_21_readvariableop_resource4
0batch_normalization_21_readvariableop_1_resourceC
?batch_normalization_21_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource?
;conv2d_transpose_3_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_3_biasadd_readvariableop_resource,
(conv2d_24_conv2d_readvariableop_resource-
)conv2d_24_biasadd_readvariableop_resource2
.batch_normalization_23_readvariableop_resource4
0batch_normalization_23_readvariableop_1_resourceC
?batch_normalization_23_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource?
;conv2d_transpose_4_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_4_biasadd_readvariableop_resource,
(conv2d_26_conv2d_readvariableop_resource-
)conv2d_26_biasadd_readvariableop_resource2
.batch_normalization_25_readvariableop_resource4
0batch_normalization_25_readvariableop_1_resourceC
?batch_normalization_25_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_25_fusedbatchnormv3_readvariableop_1_resource?
;conv2d_transpose_5_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_5_biasadd_readvariableop_resource,
(conv2d_28_conv2d_readvariableop_resource-
)conv2d_28_biasadd_readvariableop_resource2
.batch_normalization_27_readvariableop_resource4
0batch_normalization_27_readvariableop_1_resourceC
?batch_normalization_27_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_29_conv2d_readvariableop_resource-
)conv2d_29_biasadd_readvariableop_resource
identity??6batch_normalization_15/FusedBatchNormV3/ReadVariableOp?8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_15/ReadVariableOp?'batch_normalization_15/ReadVariableOp_1?6batch_normalization_17/FusedBatchNormV3/ReadVariableOp?8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_17/ReadVariableOp?'batch_normalization_17/ReadVariableOp_1?6batch_normalization_19/FusedBatchNormV3/ReadVariableOp?8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_19/ReadVariableOp?'batch_normalization_19/ReadVariableOp_1?6batch_normalization_21/FusedBatchNormV3/ReadVariableOp?8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_21/ReadVariableOp?'batch_normalization_21/ReadVariableOp_1?6batch_normalization_23/FusedBatchNormV3/ReadVariableOp?8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_23/ReadVariableOp?'batch_normalization_23/ReadVariableOp_1?6batch_normalization_25/FusedBatchNormV3/ReadVariableOp?8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_25/ReadVariableOp?'batch_normalization_25/ReadVariableOp_1?6batch_normalization_27/FusedBatchNormV3/ReadVariableOp?8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_27/ReadVariableOp?'batch_normalization_27/ReadVariableOp_1? conv2d_16/BiasAdd/ReadVariableOp?conv2d_16/Conv2D/ReadVariableOp? conv2d_18/BiasAdd/ReadVariableOp?conv2d_18/Conv2D/ReadVariableOp? conv2d_20/BiasAdd/ReadVariableOp?conv2d_20/Conv2D/ReadVariableOp? conv2d_22/BiasAdd/ReadVariableOp?conv2d_22/Conv2D/ReadVariableOp? conv2d_24/BiasAdd/ReadVariableOp?conv2d_24/Conv2D/ReadVariableOp? conv2d_26/BiasAdd/ReadVariableOp?conv2d_26/Conv2D/ReadVariableOp? conv2d_28/BiasAdd/ReadVariableOp?conv2d_28/Conv2D/ReadVariableOp? conv2d_29/BiasAdd/ReadVariableOp?conv2d_29/Conv2D/ReadVariableOp?)conv2d_transpose_3/BiasAdd/ReadVariableOp?2conv2d_transpose_3/conv2d_transpose/ReadVariableOp?)conv2d_transpose_4/BiasAdd/ReadVariableOp?2conv2d_transpose_4/conv2d_transpose/ReadVariableOp?)conv2d_transpose_5/BiasAdd/ReadVariableOp?2conv2d_transpose_5/conv2d_transpose/ReadVariableOpg
CastCastinputs*

DstT0*

SrcT0*1
_output_shapes
:???????????2
Cast?
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_16/Conv2D/ReadVariableOp?
conv2d_16/Conv2D/CastCast'conv2d_16/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: 2
conv2d_16/Conv2D/Cast?
conv2d_16/Conv2DConv2DCast:y:0conv2d_16/Conv2D/Cast:y:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
conv2d_16/Conv2D?
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_16/BiasAdd/ReadVariableOp?
conv2d_16/BiasAdd/CastCast(conv2d_16/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: 2
conv2d_16/BiasAdd/Cast?
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0conv2d_16/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:??????????? 2
conv2d_16/BiasAdd?
%batch_normalization_15/ReadVariableOpReadVariableOp.batch_normalization_15_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_15/ReadVariableOp?
'batch_normalization_15/ReadVariableOp_1ReadVariableOp0batch_normalization_15_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_15/ReadVariableOp_1?
6batch_normalization_15/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_15_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_15/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_15/FusedBatchNormV3FusedBatchNormV3conv2d_16/BiasAdd:output:0-batch_normalization_15/ReadVariableOp:value:0/batch_normalization_15/ReadVariableOp_1:value:0>batch_normalization_15/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2)
'batch_normalization_15/FusedBatchNormV3?
activation_15/ReluRelu+batch_normalization_15/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? 2
activation_15/Relu?
max_pooling2d_3/MaxPoolMaxPool activation_15/Relu:activations:0*
T0*1
_output_shapes
:??????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPool?
dropout_6/IdentityIdentity max_pooling2d_3/MaxPool:output:0*
T0*1
_output_shapes
:??????????? 2
dropout_6/Identity?
conv2d_18/Conv2D/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_18/Conv2D/ReadVariableOp?
conv2d_18/Conv2D/CastCast'conv2d_18/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: @2
conv2d_18/Conv2D/Cast?
conv2d_18/Conv2DConv2Ddropout_6/Identity:output:0conv2d_18/Conv2D/Cast:y:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv2d_18/Conv2D?
 conv2d_18/BiasAdd/ReadVariableOpReadVariableOp)conv2d_18_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_18/BiasAdd/ReadVariableOp?
conv2d_18/BiasAdd/CastCast(conv2d_18/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@2
conv2d_18/BiasAdd/Cast?
conv2d_18/BiasAddBiasAddconv2d_18/Conv2D:output:0conv2d_18/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:???????????@2
conv2d_18/BiasAdd?
%batch_normalization_17/ReadVariableOpReadVariableOp.batch_normalization_17_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_17/ReadVariableOp?
'batch_normalization_17/ReadVariableOp_1ReadVariableOp0batch_normalization_17_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_17/ReadVariableOp_1?
6batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_17/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_17/FusedBatchNormV3FusedBatchNormV3conv2d_18/BiasAdd:output:0-batch_normalization_17/ReadVariableOp:value:0/batch_normalization_17/ReadVariableOp_1:value:0>batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2)
'batch_normalization_17/FusedBatchNormV3?
activation_17/ReluRelu+batch_normalization_17/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????@2
activation_17/Relu?
max_pooling2d_4/MaxPoolMaxPool activation_17/Relu:activations:0*
T0*1
_output_shapes
:???????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_4/MaxPool?
dropout_7/IdentityIdentity max_pooling2d_4/MaxPool:output:0*
T0*1
_output_shapes
:???????????@2
dropout_7/Identity?
conv2d_20/Conv2D/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02!
conv2d_20/Conv2D/ReadVariableOp?
conv2d_20/Conv2D/CastCast'conv2d_20/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*'
_output_shapes
:@?2
conv2d_20/Conv2D/Cast?
conv2d_20/Conv2DConv2Ddropout_7/Identity:output:0conv2d_20/Conv2D/Cast:y:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
conv2d_20/Conv2D?
 conv2d_20/BiasAdd/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_20/BiasAdd/ReadVariableOp?
conv2d_20/BiasAdd/CastCast(conv2d_20/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:?2
conv2d_20/BiasAdd/Cast?
conv2d_20/BiasAddBiasAddconv2d_20/Conv2D:output:0conv2d_20/BiasAdd/Cast:y:0*
T0*2
_output_shapes 
:????????????2
conv2d_20/BiasAdd?
%batch_normalization_19/ReadVariableOpReadVariableOp.batch_normalization_19_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_19/ReadVariableOp?
'batch_normalization_19/ReadVariableOp_1ReadVariableOp0batch_normalization_19_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_19/ReadVariableOp_1?
6batch_normalization_19/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_19_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_19/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_19/FusedBatchNormV3FusedBatchNormV3conv2d_20/BiasAdd:output:0-batch_normalization_19/ReadVariableOp:value:0/batch_normalization_19/ReadVariableOp_1:value:0>batch_normalization_19/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2)
'batch_normalization_19/FusedBatchNormV3?
activation_19/ReluRelu+batch_normalization_19/FusedBatchNormV3:y:0*
T0*2
_output_shapes 
:????????????2
activation_19/Relu?
max_pooling2d_5/MaxPoolMaxPool activation_19/Relu:activations:0*
T0*0
_output_shapes
:?????????D\?*
ksize
*
paddingVALID*
strides
2
max_pooling2d_5/MaxPool?
dropout_8/IdentityIdentity max_pooling2d_5/MaxPool:output:0*
T0*0
_output_shapes
:?????????D\?2
dropout_8/Identity?
conv2d_22/Conv2D/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_22/Conv2D/ReadVariableOp?
conv2d_22/Conv2D/CastCast'conv2d_22/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*(
_output_shapes
:??2
conv2d_22/Conv2D/Cast?
conv2d_22/Conv2DConv2Ddropout_8/Identity:output:0conv2d_22/Conv2D/Cast:y:0*
T0*0
_output_shapes
:?????????D\?*
paddingSAME*
strides
2
conv2d_22/Conv2D?
 conv2d_22/BiasAdd/ReadVariableOpReadVariableOp)conv2d_22_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_22/BiasAdd/ReadVariableOp?
conv2d_22/BiasAdd/CastCast(conv2d_22/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:?2
conv2d_22/BiasAdd/Cast?
conv2d_22/BiasAddBiasAddconv2d_22/Conv2D:output:0conv2d_22/BiasAdd/Cast:y:0*
T0*0
_output_shapes
:?????????D\?2
conv2d_22/BiasAdd?
%batch_normalization_21/ReadVariableOpReadVariableOp.batch_normalization_21_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_21/ReadVariableOp?
'batch_normalization_21/ReadVariableOp_1ReadVariableOp0batch_normalization_21_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_21/ReadVariableOp_1?
6batch_normalization_21/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_21_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_21/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_21/FusedBatchNormV3FusedBatchNormV3conv2d_22/BiasAdd:output:0-batch_normalization_21/ReadVariableOp:value:0/batch_normalization_21/ReadVariableOp_1:value:0>batch_normalization_21/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????D\?:?:?:?:?:*
epsilon%o?:*
is_training( 2)
'batch_normalization_21/FusedBatchNormV3?
activation_21/ReluRelu+batch_normalization_21/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????D\?2
activation_21/Relu?
conv2d_transpose_3/ShapeShape activation_21/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_3/Shape?
&conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_3/strided_slice/stack?
(conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_3/strided_slice/stack_1?
(conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_3/strided_slice/stack_2?
 conv2d_transpose_3/strided_sliceStridedSlice!conv2d_transpose_3/Shape:output:0/conv2d_transpose_3/strided_slice/stack:output:01conv2d_transpose_3/strided_slice/stack_1:output:01conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_3/strided_slice{
conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_3/stack/1{
conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_3/stack/2{
conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_3/stack/3?
conv2d_transpose_3/stackPack)conv2d_transpose_3/strided_slice:output:0#conv2d_transpose_3/stack/1:output:0#conv2d_transpose_3/stack/2:output:0#conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_3/stack?
(conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_3/strided_slice_1/stack?
*conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_1/stack_1?
*conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_1/stack_2?
"conv2d_transpose_3/strided_slice_1StridedSlice!conv2d_transpose_3/stack:output:01conv2d_transpose_3/strided_slice_1/stack:output:03conv2d_transpose_3/strided_slice_1/stack_1:output:03conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_3/strided_slice_1?
2conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_3_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype024
2conv2d_transpose_3/conv2d_transpose/ReadVariableOp?
(conv2d_transpose_3/conv2d_transpose/CastCast:conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0*

DstT0*

SrcT0*(
_output_shapes
:??2*
(conv2d_transpose_3/conv2d_transpose/Cast?
#conv2d_transpose_3/conv2d_transposeConv2DBackpropInput!conv2d_transpose_3/stack:output:0,conv2d_transpose_3/conv2d_transpose/Cast:y:0 activation_21/Relu:activations:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2%
#conv2d_transpose_3/conv2d_transpose?
)conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)conv2d_transpose_3/BiasAdd/ReadVariableOp?
conv2d_transpose_3/BiasAdd/CastCast1conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:?2!
conv2d_transpose_3/BiasAdd/Cast?
conv2d_transpose_3/BiasAddBiasAdd,conv2d_transpose_3/conv2d_transpose:output:0#conv2d_transpose_3/BiasAdd/Cast:y:0*
T0*2
_output_shapes 
:????????????2
conv2d_transpose_3/BiasAddx
concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_3/concat/axis?
concatenate_3/concatConcatV2#conv2d_transpose_3/BiasAdd:output:0 activation_19/Relu:activations:0"concatenate_3/concat/axis:output:0*
N*
T0*2
_output_shapes 
:????????????2
concatenate_3/concat?
dropout_9/IdentityIdentityconcatenate_3/concat:output:0*
T0*2
_output_shapes 
:????????????2
dropout_9/Identity?
conv2d_24/Conv2D/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_24/Conv2D/ReadVariableOp?
conv2d_24/Conv2D/CastCast'conv2d_24/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*(
_output_shapes
:??2
conv2d_24/Conv2D/Cast?
conv2d_24/Conv2DConv2Ddropout_9/Identity:output:0conv2d_24/Conv2D/Cast:y:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
conv2d_24/Conv2D?
 conv2d_24/BiasAdd/ReadVariableOpReadVariableOp)conv2d_24_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_24/BiasAdd/ReadVariableOp?
conv2d_24/BiasAdd/CastCast(conv2d_24/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:?2
conv2d_24/BiasAdd/Cast?
conv2d_24/BiasAddBiasAddconv2d_24/Conv2D:output:0conv2d_24/BiasAdd/Cast:y:0*
T0*2
_output_shapes 
:????????????2
conv2d_24/BiasAdd?
%batch_normalization_23/ReadVariableOpReadVariableOp.batch_normalization_23_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_23/ReadVariableOp?
'batch_normalization_23/ReadVariableOp_1ReadVariableOp0batch_normalization_23_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_23/ReadVariableOp_1?
6batch_normalization_23/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_23_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_23/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_23/FusedBatchNormV3FusedBatchNormV3conv2d_24/BiasAdd:output:0-batch_normalization_23/ReadVariableOp:value:0/batch_normalization_23/ReadVariableOp_1:value:0>batch_normalization_23/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2)
'batch_normalization_23/FusedBatchNormV3?
activation_23/ReluRelu+batch_normalization_23/FusedBatchNormV3:y:0*
T0*2
_output_shapes 
:????????????2
activation_23/Relu?
conv2d_transpose_4/ShapeShape activation_23/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_4/Shape?
&conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_4/strided_slice/stack?
(conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_4/strided_slice/stack_1?
(conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_4/strided_slice/stack_2?
 conv2d_transpose_4/strided_sliceStridedSlice!conv2d_transpose_4/Shape:output:0/conv2d_transpose_4/strided_slice/stack:output:01conv2d_transpose_4/strided_slice/stack_1:output:01conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_4/strided_slice{
conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_4/stack/1{
conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_4/stack/2z
conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_4/stack/3?
conv2d_transpose_4/stackPack)conv2d_transpose_4/strided_slice:output:0#conv2d_transpose_4/stack/1:output:0#conv2d_transpose_4/stack/2:output:0#conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_4/stack?
(conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_4/strided_slice_1/stack?
*conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_4/strided_slice_1/stack_1?
*conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_4/strided_slice_1/stack_2?
"conv2d_transpose_4/strided_slice_1StridedSlice!conv2d_transpose_4/stack:output:01conv2d_transpose_4/strided_slice_1/stack:output:03conv2d_transpose_4/strided_slice_1/stack_1:output:03conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_4/strided_slice_1?
2conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_4_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype024
2conv2d_transpose_4/conv2d_transpose/ReadVariableOp?
(conv2d_transpose_4/conv2d_transpose/CastCast:conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0*

DstT0*

SrcT0*'
_output_shapes
:@?2*
(conv2d_transpose_4/conv2d_transpose/Cast?
#conv2d_transpose_4/conv2d_transposeConv2DBackpropInput!conv2d_transpose_4/stack:output:0,conv2d_transpose_4/conv2d_transpose/Cast:y:0 activation_23/Relu:activations:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2%
#conv2d_transpose_4/conv2d_transpose?
)conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)conv2d_transpose_4/BiasAdd/ReadVariableOp?
conv2d_transpose_4/BiasAdd/CastCast1conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@2!
conv2d_transpose_4/BiasAdd/Cast?
conv2d_transpose_4/BiasAddBiasAdd,conv2d_transpose_4/conv2d_transpose:output:0#conv2d_transpose_4/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:???????????@2
conv2d_transpose_4/BiasAddx
concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_4/concat/axis?
concatenate_4/concatConcatV2#conv2d_transpose_4/BiasAdd:output:0 activation_17/Relu:activations:0"concatenate_4/concat/axis:output:0*
N*
T0*2
_output_shapes 
:????????????2
concatenate_4/concat?
dropout_10/IdentityIdentityconcatenate_4/concat:output:0*
T0*2
_output_shapes 
:????????????2
dropout_10/Identity?
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02!
conv2d_26/Conv2D/ReadVariableOp?
conv2d_26/Conv2D/CastCast'conv2d_26/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*'
_output_shapes
:?@2
conv2d_26/Conv2D/Cast?
conv2d_26/Conv2DConv2Ddropout_10/Identity:output:0conv2d_26/Conv2D/Cast:y:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv2d_26/Conv2D?
 conv2d_26/BiasAdd/ReadVariableOpReadVariableOp)conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_26/BiasAdd/ReadVariableOp?
conv2d_26/BiasAdd/CastCast(conv2d_26/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@2
conv2d_26/BiasAdd/Cast?
conv2d_26/BiasAddBiasAddconv2d_26/Conv2D:output:0conv2d_26/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:???????????@2
conv2d_26/BiasAdd?
%batch_normalization_25/ReadVariableOpReadVariableOp.batch_normalization_25_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_25/ReadVariableOp?
'batch_normalization_25/ReadVariableOp_1ReadVariableOp0batch_normalization_25_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_25/ReadVariableOp_1?
6batch_normalization_25/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_25_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_25/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_25_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_25/FusedBatchNormV3FusedBatchNormV3conv2d_26/BiasAdd:output:0-batch_normalization_25/ReadVariableOp:value:0/batch_normalization_25/ReadVariableOp_1:value:0>batch_normalization_25/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2)
'batch_normalization_25/FusedBatchNormV3?
activation_25/ReluRelu+batch_normalization_25/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????@2
activation_25/Relu?
conv2d_transpose_5/ShapeShape activation_25/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_5/Shape?
&conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_5/strided_slice/stack?
(conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_5/strided_slice/stack_1?
(conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_5/strided_slice/stack_2?
 conv2d_transpose_5/strided_sliceStridedSlice!conv2d_transpose_5/Shape:output:0/conv2d_transpose_5/strided_slice/stack:output:01conv2d_transpose_5/strided_slice/stack_1:output:01conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_5/strided_slice{
conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_5/stack/1{
conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_5/stack/2z
conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_5/stack/3?
conv2d_transpose_5/stackPack)conv2d_transpose_5/strided_slice:output:0#conv2d_transpose_5/stack/1:output:0#conv2d_transpose_5/stack/2:output:0#conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_5/stack?
(conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_5/strided_slice_1/stack?
*conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_5/strided_slice_1/stack_1?
*conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_5/strided_slice_1/stack_2?
"conv2d_transpose_5/strided_slice_1StridedSlice!conv2d_transpose_5/stack:output:01conv2d_transpose_5/strided_slice_1/stack:output:03conv2d_transpose_5/strided_slice_1/stack_1:output:03conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_5/strided_slice_1?
2conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype024
2conv2d_transpose_5/conv2d_transpose/ReadVariableOp?
(conv2d_transpose_5/conv2d_transpose/CastCast:conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: @2*
(conv2d_transpose_5/conv2d_transpose/Cast?
#conv2d_transpose_5/conv2d_transposeConv2DBackpropInput!conv2d_transpose_5/stack:output:0,conv2d_transpose_5/conv2d_transpose/Cast:y:0 activation_25/Relu:activations:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2%
#conv2d_transpose_5/conv2d_transpose?
)conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)conv2d_transpose_5/BiasAdd/ReadVariableOp?
conv2d_transpose_5/BiasAdd/CastCast1conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: 2!
conv2d_transpose_5/BiasAdd/Cast?
conv2d_transpose_5/BiasAddBiasAdd,conv2d_transpose_5/conv2d_transpose:output:0#conv2d_transpose_5/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:??????????? 2
conv2d_transpose_5/BiasAddx
concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_5/concat/axis?
concatenate_5/concatConcatV2#conv2d_transpose_5/BiasAdd:output:0 activation_15/Relu:activations:0"concatenate_5/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????@2
concatenate_5/concat?
dropout_11/IdentityIdentityconcatenate_5/concat:output:0*
T0*1
_output_shapes
:???????????@2
dropout_11/Identity?
conv2d_28/Conv2D/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02!
conv2d_28/Conv2D/ReadVariableOp?
conv2d_28/Conv2D/CastCast'conv2d_28/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:@ 2
conv2d_28/Conv2D/Cast?
conv2d_28/Conv2DConv2Ddropout_11/Identity:output:0conv2d_28/Conv2D/Cast:y:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
conv2d_28/Conv2D?
 conv2d_28/BiasAdd/ReadVariableOpReadVariableOp)conv2d_28_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_28/BiasAdd/ReadVariableOp?
conv2d_28/BiasAdd/CastCast(conv2d_28/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: 2
conv2d_28/BiasAdd/Cast?
conv2d_28/BiasAddBiasAddconv2d_28/Conv2D:output:0conv2d_28/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:??????????? 2
conv2d_28/BiasAdd?
%batch_normalization_27/ReadVariableOpReadVariableOp.batch_normalization_27_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_27/ReadVariableOp?
'batch_normalization_27/ReadVariableOp_1ReadVariableOp0batch_normalization_27_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_27/ReadVariableOp_1?
6batch_normalization_27/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_27_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_27/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_27/FusedBatchNormV3FusedBatchNormV3conv2d_28/BiasAdd:output:0-batch_normalization_27/ReadVariableOp:value:0/batch_normalization_27/ReadVariableOp_1:value:0>batch_normalization_27/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2)
'batch_normalization_27/FusedBatchNormV3?
activation_27/ReluRelu+batch_normalization_27/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? 2
activation_27/Relu?
conv2d_29/Conv2D/ReadVariableOpReadVariableOp(conv2d_29_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_29/Conv2D/ReadVariableOp?
conv2d_29/Conv2D/CastCast'conv2d_29/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: 2
conv2d_29/Conv2D/Cast?
conv2d_29/Conv2DConv2D activation_27/Relu:activations:0conv2d_29/Conv2D/Cast:y:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
conv2d_29/Conv2D?
 conv2d_29/BiasAdd/ReadVariableOpReadVariableOp)conv2d_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_29/BiasAdd/ReadVariableOp?
conv2d_29/BiasAdd/CastCast(conv2d_29/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:2
conv2d_29/BiasAdd/Cast?
conv2d_29/BiasAddBiasAddconv2d_29/Conv2D:output:0conv2d_29/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:???????????2
conv2d_29/BiasAdd?
conv2d_29/SigmoidSigmoidconv2d_29/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_29/Sigmoid?
IdentityIdentityconv2d_29/Sigmoid:y:07^batch_normalization_15/FusedBatchNormV3/ReadVariableOp9^batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_15/ReadVariableOp(^batch_normalization_15/ReadVariableOp_17^batch_normalization_17/FusedBatchNormV3/ReadVariableOp9^batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_17/ReadVariableOp(^batch_normalization_17/ReadVariableOp_17^batch_normalization_19/FusedBatchNormV3/ReadVariableOp9^batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_19/ReadVariableOp(^batch_normalization_19/ReadVariableOp_17^batch_normalization_21/FusedBatchNormV3/ReadVariableOp9^batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_21/ReadVariableOp(^batch_normalization_21/ReadVariableOp_17^batch_normalization_23/FusedBatchNormV3/ReadVariableOp9^batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_23/ReadVariableOp(^batch_normalization_23/ReadVariableOp_17^batch_normalization_25/FusedBatchNormV3/ReadVariableOp9^batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_25/ReadVariableOp(^batch_normalization_25/ReadVariableOp_17^batch_normalization_27/FusedBatchNormV3/ReadVariableOp9^batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_27/ReadVariableOp(^batch_normalization_27/ReadVariableOp_1!^conv2d_16/BiasAdd/ReadVariableOp ^conv2d_16/Conv2D/ReadVariableOp!^conv2d_18/BiasAdd/ReadVariableOp ^conv2d_18/Conv2D/ReadVariableOp!^conv2d_20/BiasAdd/ReadVariableOp ^conv2d_20/Conv2D/ReadVariableOp!^conv2d_22/BiasAdd/ReadVariableOp ^conv2d_22/Conv2D/ReadVariableOp!^conv2d_24/BiasAdd/ReadVariableOp ^conv2d_24/Conv2D/ReadVariableOp!^conv2d_26/BiasAdd/ReadVariableOp ^conv2d_26/Conv2D/ReadVariableOp!^conv2d_28/BiasAdd/ReadVariableOp ^conv2d_28/Conv2D/ReadVariableOp!^conv2d_29/BiasAdd/ReadVariableOp ^conv2d_29/Conv2D/ReadVariableOp*^conv2d_transpose_3/BiasAdd/ReadVariableOp3^conv2d_transpose_3/conv2d_transpose/ReadVariableOp*^conv2d_transpose_4/BiasAdd/ReadVariableOp3^conv2d_transpose_4/conv2d_transpose/ReadVariableOp*^conv2d_transpose_5/BiasAdd/ReadVariableOp3^conv2d_transpose_5/conv2d_transpose/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::::::::::2p
6batch_normalization_15/FusedBatchNormV3/ReadVariableOp6batch_normalization_15/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_18batch_normalization_15/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_15/ReadVariableOp%batch_normalization_15/ReadVariableOp2R
'batch_normalization_15/ReadVariableOp_1'batch_normalization_15/ReadVariableOp_12p
6batch_normalization_17/FusedBatchNormV3/ReadVariableOp6batch_normalization_17/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_18batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_17/ReadVariableOp%batch_normalization_17/ReadVariableOp2R
'batch_normalization_17/ReadVariableOp_1'batch_normalization_17/ReadVariableOp_12p
6batch_normalization_19/FusedBatchNormV3/ReadVariableOp6batch_normalization_19/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_18batch_normalization_19/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_19/ReadVariableOp%batch_normalization_19/ReadVariableOp2R
'batch_normalization_19/ReadVariableOp_1'batch_normalization_19/ReadVariableOp_12p
6batch_normalization_21/FusedBatchNormV3/ReadVariableOp6batch_normalization_21/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_18batch_normalization_21/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_21/ReadVariableOp%batch_normalization_21/ReadVariableOp2R
'batch_normalization_21/ReadVariableOp_1'batch_normalization_21/ReadVariableOp_12p
6batch_normalization_23/FusedBatchNormV3/ReadVariableOp6batch_normalization_23/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_18batch_normalization_23/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_23/ReadVariableOp%batch_normalization_23/ReadVariableOp2R
'batch_normalization_23/ReadVariableOp_1'batch_normalization_23/ReadVariableOp_12p
6batch_normalization_25/FusedBatchNormV3/ReadVariableOp6batch_normalization_25/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_18batch_normalization_25/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_25/ReadVariableOp%batch_normalization_25/ReadVariableOp2R
'batch_normalization_25/ReadVariableOp_1'batch_normalization_25/ReadVariableOp_12p
6batch_normalization_27/FusedBatchNormV3/ReadVariableOp6batch_normalization_27/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_18batch_normalization_27/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_27/ReadVariableOp%batch_normalization_27/ReadVariableOp2R
'batch_normalization_27/ReadVariableOp_1'batch_normalization_27/ReadVariableOp_12D
 conv2d_16/BiasAdd/ReadVariableOp conv2d_16/BiasAdd/ReadVariableOp2B
conv2d_16/Conv2D/ReadVariableOpconv2d_16/Conv2D/ReadVariableOp2D
 conv2d_18/BiasAdd/ReadVariableOp conv2d_18/BiasAdd/ReadVariableOp2B
conv2d_18/Conv2D/ReadVariableOpconv2d_18/Conv2D/ReadVariableOp2D
 conv2d_20/BiasAdd/ReadVariableOp conv2d_20/BiasAdd/ReadVariableOp2B
conv2d_20/Conv2D/ReadVariableOpconv2d_20/Conv2D/ReadVariableOp2D
 conv2d_22/BiasAdd/ReadVariableOp conv2d_22/BiasAdd/ReadVariableOp2B
conv2d_22/Conv2D/ReadVariableOpconv2d_22/Conv2D/ReadVariableOp2D
 conv2d_24/BiasAdd/ReadVariableOp conv2d_24/BiasAdd/ReadVariableOp2B
conv2d_24/Conv2D/ReadVariableOpconv2d_24/Conv2D/ReadVariableOp2D
 conv2d_26/BiasAdd/ReadVariableOp conv2d_26/BiasAdd/ReadVariableOp2B
conv2d_26/Conv2D/ReadVariableOpconv2d_26/Conv2D/ReadVariableOp2D
 conv2d_28/BiasAdd/ReadVariableOp conv2d_28/BiasAdd/ReadVariableOp2B
conv2d_28/Conv2D/ReadVariableOpconv2d_28/Conv2D/ReadVariableOp2D
 conv2d_29/BiasAdd/ReadVariableOp conv2d_29/BiasAdd/ReadVariableOp2B
conv2d_29/Conv2D/ReadVariableOpconv2d_29/Conv2D/ReadVariableOp2V
)conv2d_transpose_3/BiasAdd/ReadVariableOp)conv2d_transpose_3/BiasAdd/ReadVariableOp2h
2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_4/BiasAdd/ReadVariableOp)conv2d_transpose_4/BiasAdd/ReadVariableOp2h
2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_5/BiasAdd/ReadVariableOp)conv2d_transpose_5/BiasAdd/ReadVariableOp2h
2conv2d_transpose_5/conv2d_transpose/ReadVariableOp2conv2d_transpose_5/conv2d_transpose/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_25_layer_call_fn_42127

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_25_layer_call_and_return_conditional_losses_392492
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
r
H__inference_concatenate_3_layer_call_and_return_conditional_losses_38998

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*2
_output_shapes 
:????????????2
concatn
IdentityIdentityconcat:output:0*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:,????????????????????????????:????????????:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs:ZV
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_15_layer_call_fn_41171

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_384842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:??????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
I
-__inference_activation_15_layer_call_fn_41194

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *Q
fLRJ
H__inference_activation_15_layer_call_and_return_conditional_losses_385432
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
D__inference_conv2d_22_layer_call_and_return_conditional_losses_41605

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*(
_output_shapes
:??2
Conv2D/Cast?
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*0
_output_shapes
:?????????D\?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:?2
BiasAdd/Cast{
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*0
_output_shapes
:?????????D\?2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????D\?2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????D\?::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????D\?
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_41280

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_17_layer_call_fn_41357

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_386292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
Y
-__inference_concatenate_3_layer_call_fn_41765
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *Q
fLRJ
H__inference_concatenate_3_layer_call_and_return_conditional_losses_389982
PartitionedCallw
IdentityIdentityPartitionedCall:output:0*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:,????????????????????????????:????????????:l h
B
_output_shapes0
.:,????????????????????????????
"
_user_specified_name
inputs/0:\X
2
_output_shapes 
:????????????
"
_user_specified_name
inputs/1
?
d
E__inference_dropout_10_layer_call_and_return_conditional_losses_41976

inputs
identity?a
dropout/ConstConst*
_output_shapes
: *
dtype0*
value
B j?x2
dropout/Const~
dropout/MulMulinputsdropout/Const:output:0*
T0*2
_output_shapes 
:????????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*2
_output_shapes 
:????????????*
dtype02&
$dropout/random_uniform/RandomUniforms
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value
B j?\2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*2
_output_shapes 
:????????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*2
_output_shapes 
:????????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*2
_output_shapes 
:????????????2
dropout/Mul_1p
IdentityIdentitydropout/Mul_1:z:0*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*1
_input_shapes 
:????????????:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_21_layer_call_and_return_conditional_losses_37971

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
d
H__inference_activation_19_layer_call_and_return_conditional_losses_38833

inputs
identityY
ReluReluinputs*
T0*2
_output_shapes 
:????????????2
Reluq
IdentityIdentityRelu:activations:0*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*1
_input_shapes 
:????????????:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_21_layer_call_and_return_conditional_losses_38937

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????D\?:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:?????????D\?2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????D\?::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????D\?
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_23_layer_call_and_return_conditional_losses_39102

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
r
H__inference_concatenate_4_layer_call_and_return_conditional_losses_39163

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*2
_output_shapes 
:????????????2
concatn
IdentityIdentityconcat:output:0*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:+???????????????????????????@:???????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs:YU
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
b
D__inference_dropout_6_layer_call_and_return_conditional_losses_41211

inputs

identity_1d
IdentityIdentityinputs*
T0*1
_output_shapes
:??????????? 2

Identitys

Identity_1IdentityIdentity:output:0*
T0*1
_output_shapes
:??????????? 2

Identity_1"!

identity_1Identity_1:output:0*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_19_layer_call_fn_41479

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_378242
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_21_layer_call_and_return_conditional_losses_37940

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
r
H__inference_concatenate_5_layer_call_and_return_conditional_losses_39328

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????@2
concatm
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:+??????????????????????????? :??????????? :i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs:YU
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_23_layer_call_fn_41941

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_23_layer_call_and_return_conditional_losses_391022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
F
*__inference_dropout_11_layer_call_fn_42190

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *N
fIRG
E__inference_dropout_11_layer_call_and_return_conditional_losses_393542
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????@:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?%
?
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_38018

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1U
stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transpose/CastCast'conv2d_transpose/ReadVariableOp:value:0*

DstT0*

SrcT0*(
_output_shapes
:??2
conv2d_transpose/Cast?
conv2d_transposeConv2DBackpropInputstack:output:0conv2d_transpose/Cast:y:0inputs*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:?2
BiasAdd/Cast?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/Cast:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
'__inference_model_1_layer_call_fn_40930

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*F
_read_only_resource_inputs(
&$	
!"#$%&)*+,-.12*8
config_proto(&

CPU

GPU2*0J

  ?E8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_397982
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
b
)__inference_dropout_7_layer_call_fn_41402

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_387092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????@22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
b
D__inference_dropout_9_layer_call_and_return_conditional_losses_39024

inputs

identity_1e
IdentityIdentityinputs*
T0*2
_output_shapes 
:????????????2

Identityt

Identity_1IdentityIdentity:output:0*
T0*2
_output_shapes 
:????????????2

Identity_1"!

identity_1Identity_1:output:0*1
_input_shapes 
:????????????:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
c
D__inference_dropout_6_layer_call_and_return_conditional_losses_38564

inputs
identity?a
dropout/ConstConst*
_output_shapes
: *
dtype0*
value
B j?x2
dropout/Const}
dropout/MulMulinputsdropout/Const:output:0*
T0*1
_output_shapes
:??????????? 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*1
_output_shapes
:??????????? *
dtype02&
$dropout/random_uniform/RandomUniforms
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value
B j?\2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:??????????? 2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:??????????? 2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*1
_output_shapes
:??????????? 2
dropout/Mul_1o
IdentityIdentitydropout/Mul_1:z:0*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_38484

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:??????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
c
D__inference_dropout_8_layer_call_and_return_conditional_losses_41578

inputs
identity?a
dropout/ConstConst*
_output_shapes
: *
dtype0*
value
B j?x2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:?????????D\?2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:?????????D\?*
dtype02&
$dropout/random_uniform/RandomUniforms
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value
B j?\2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:?????????D\?2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:?????????D\?2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:?????????D\?2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:?????????D\?2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????D\?:X T
0
_output_shapes
:?????????D\?
 
_user_specified_nameinputs
?
b
)__inference_dropout_8_layer_call_fn_41588

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????D\?* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_388542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????D\?2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????D\?22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????D\?
 
_user_specified_nameinputs
?
d
H__inference_activation_17_layer_call_and_return_conditional_losses_38688

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:???????????@2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????@:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
c
E__inference_dropout_11_layer_call_and_return_conditional_losses_39354

inputs

identity_1d
IdentityIdentityinputs*
T0*1
_output_shapes
:???????????@2

Identitys

Identity_1IdentityIdentity:output:0*
T0*1
_output_shapes
:???????????@2

Identity_1"!

identity_1Identity_1:output:0*0
_input_shapes
:???????????@:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
c
E__inference_dropout_10_layer_call_and_return_conditional_losses_39189

inputs

identity_1e
IdentityIdentityinputs*
T0*2
_output_shapes 
:????????????2

Identityt

Identity_1IdentityIdentity:output:0*
T0*2
_output_shapes 
:????????????2

Identity_1"!

identity_1Identity_1:output:0*1
_input_shapes 
:????????????:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
I
-__inference_activation_23_layer_call_fn_41951

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *Q
fLRJ
H__inference_activation_23_layer_call_and_return_conditional_losses_391432
PartitionedCallw
IdentityIdentityPartitionedCall:output:0*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*1
_input_shapes 
:????????????:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_37623

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
D__inference_conv2d_16_layer_call_and_return_conditional_losses_41047

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: 2
Conv2D/Cast?
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpx
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: 2
BiasAdd/Cast|
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*1
_output_shapes
:??????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_19_layer_call_fn_41556

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_387922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_25_layer_call_and_return_conditional_losses_42032

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
~
)__inference_conv2d_24_layer_call_fn_41813

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *M
fHRF
D__inference_conv2d_24_layer_call_and_return_conditional_losses_390492
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:????????????::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_37640

inputs
identity?
MaxPoolMaxPoolinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_38774

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
D__inference_conv2d_22_layer_call_and_return_conditional_losses_38884

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*(
_output_shapes
:??2
Conv2D/Cast?
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*0
_output_shapes
:?????????D\?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:?2
BiasAdd/Cast{
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*0
_output_shapes
:?????????D\?2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????D\?2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????D\?::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????D\?
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_41158

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:??????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_27_layer_call_fn_42339

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_27_layer_call_and_return_conditional_losses_384212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_21_layer_call_fn_41665

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_21_layer_call_and_return_conditional_losses_379402
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
b
D__inference_dropout_8_layer_call_and_return_conditional_losses_41583

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:?????????D\?2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:?????????D\?2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:?????????D\?:X T
0
_output_shapes
:?????????D\?
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_41466

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_37855

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
D__inference_conv2d_26_layer_call_and_return_conditional_losses_39214

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02
Conv2D/ReadVariableOp?
Conv2D/CastCastConv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*'
_output_shapes
:?@2
Conv2D/Cast?
Conv2DConv2DinputsConv2D/Cast:y:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpx
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@2
BiasAdd/Cast|
BiasAddBiasAddConv2D:output:0BiasAdd/Cast:y:0*
T0*1
_output_shapes
:???????????@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
'__inference_model_1_layer_call_fn_41035

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./012*8
config_proto(&

CPU

GPU2*0J

  ?E8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_400452
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
I
-__inference_activation_21_layer_call_fn_41752

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????D\?* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *Q
fLRJ
H__inference_activation_21_layer_call_and_return_conditional_losses_389782
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????D\?2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????D\?:X T
0
_output_shapes
:?????????D\?
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_19_layer_call_fn_41543

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *Z
fURS
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_387742
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*2
_output_shapes 
:????????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
E
input_2:
serving_default_input_2:0???????????G
	conv2d_29:
StatefulPartitionedCall:0???????????tensorflow/serving/predict:??	
??
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer-8

layer-9
layer-10
layer_with_weights-4
layer-11
layer_with_weights-5
layer-12
layer-13
layer-14
layer-15
layer_with_weights-6
layer-16
layer_with_weights-7
layer-17
layer-18
layer_with_weights-8
layer-19
layer-20
layer-21
layer_with_weights-9
layer-22
layer_with_weights-10
layer-23
layer-24
layer_with_weights-11
layer-25
layer-26
layer-27
layer_with_weights-12
layer-28
layer_with_weights-13
layer-29
layer-30
 layer_with_weights-14
 layer-31
!layer-32
"layer-33
#layer_with_weights-15
#layer-34
$layer_with_weights-16
$layer-35
%layer-36
&layer_with_weights-17
&layer-37
'	optimizer
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,
signatures
+?&call_and_return_all_conditional_losses
?_default_save_signature
?__call__"??
_tf_keras_network??{"class_name": "Functional", "name": "model_1", "trainable": true, "expects_training_arg": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 544, 736, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_16", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_16", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_15", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_15", "inbound_nodes": [[["conv2d_16", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_15", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "activation": "relu"}, "name": "activation_15", "inbound_nodes": [[["batch_normalization_15", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_3", "inbound_nodes": [[["activation_15", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_6", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_6", "inbound_nodes": [[["max_pooling2d_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_18", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_18", "inbound_nodes": [[["dropout_6", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_17", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_17", "inbound_nodes": [[["conv2d_18", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_17", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "activation": "relu"}, "name": "activation_17", "inbound_nodes": [[["batch_normalization_17", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_4", "inbound_nodes": [[["activation_17", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_7", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_7", "inbound_nodes": [[["max_pooling2d_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_20", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_20", "inbound_nodes": [[["dropout_7", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_19", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_19", "inbound_nodes": [[["conv2d_20", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_19", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "activation": "relu"}, "name": "activation_19", "inbound_nodes": [[["batch_normalization_19", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_5", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_5", "inbound_nodes": [[["activation_19", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_8", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_8", "inbound_nodes": [[["max_pooling2d_5", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_22", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_22", "inbound_nodes": [[["dropout_8", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_21", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_21", "inbound_nodes": [[["conv2d_22", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_21", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "activation": "relu"}, "name": "activation_21", "inbound_nodes": [[["batch_normalization_21", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_3", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_3", "inbound_nodes": [[["activation_21", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_3", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": -1}, "name": "concatenate_3", "inbound_nodes": [[["conv2d_transpose_3", 0, 0, {}], ["activation_19", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_9", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_9", "inbound_nodes": [[["concatenate_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_24", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_24", "inbound_nodes": [[["dropout_9", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_23", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_23", "inbound_nodes": [[["conv2d_24", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_23", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "activation": "relu"}, "name": "activation_23", "inbound_nodes": [[["batch_normalization_23", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_4", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_4", "inbound_nodes": [[["activation_23", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_4", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": -1}, "name": "concatenate_4", "inbound_nodes": [[["conv2d_transpose_4", 0, 0, {}], ["activation_17", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_10", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_10", "inbound_nodes": [[["concatenate_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_26", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_26", "inbound_nodes": [[["dropout_10", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_25", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_25", "inbound_nodes": [[["conv2d_26", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_25", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "activation": "relu"}, "name": "activation_25", "inbound_nodes": [[["batch_normalization_25", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_5", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_5", "inbound_nodes": [[["activation_25", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_5", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": -1}, "name": "concatenate_5", "inbound_nodes": [[["conv2d_transpose_5", 0, 0, {}], ["activation_15", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_11", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_11", "inbound_nodes": [[["concatenate_5", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_28", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_28", "inbound_nodes": [[["dropout_11", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_27", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_27", "inbound_nodes": [[["conv2d_28", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_27", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "activation": "relu"}, "name": "activation_27", "inbound_nodes": [[["batch_normalization_27", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_29", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 7, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_29", "inbound_nodes": [[["activation_27", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["conv2d_29", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 544, 736, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 544, 736, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 544, 736, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_16", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_16", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_15", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_15", "inbound_nodes": [[["conv2d_16", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_15", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "activation": "relu"}, "name": "activation_15", "inbound_nodes": [[["batch_normalization_15", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_3", "inbound_nodes": [[["activation_15", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_6", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_6", "inbound_nodes": [[["max_pooling2d_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_18", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_18", "inbound_nodes": [[["dropout_6", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_17", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_17", "inbound_nodes": [[["conv2d_18", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_17", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "activation": "relu"}, "name": "activation_17", "inbound_nodes": [[["batch_normalization_17", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_4", "inbound_nodes": [[["activation_17", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_7", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_7", "inbound_nodes": [[["max_pooling2d_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_20", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_20", "inbound_nodes": [[["dropout_7", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_19", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_19", "inbound_nodes": [[["conv2d_20", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_19", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "activation": "relu"}, "name": "activation_19", "inbound_nodes": [[["batch_normalization_19", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_5", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_5", "inbound_nodes": [[["activation_19", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_8", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_8", "inbound_nodes": [[["max_pooling2d_5", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_22", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_22", "inbound_nodes": [[["dropout_8", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_21", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_21", "inbound_nodes": [[["conv2d_22", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_21", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "activation": "relu"}, "name": "activation_21", "inbound_nodes": [[["batch_normalization_21", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_3", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_3", "inbound_nodes": [[["activation_21", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_3", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": -1}, "name": "concatenate_3", "inbound_nodes": [[["conv2d_transpose_3", 0, 0, {}], ["activation_19", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_9", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_9", "inbound_nodes": [[["concatenate_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_24", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_24", "inbound_nodes": [[["dropout_9", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_23", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_23", "inbound_nodes": [[["conv2d_24", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_23", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "activation": "relu"}, "name": "activation_23", "inbound_nodes": [[["batch_normalization_23", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_4", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_4", "inbound_nodes": [[["activation_23", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_4", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": -1}, "name": "concatenate_4", "inbound_nodes": [[["conv2d_transpose_4", 0, 0, {}], ["activation_17", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_10", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_10", "inbound_nodes": [[["concatenate_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_26", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_26", "inbound_nodes": [[["dropout_10", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_25", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_25", "inbound_nodes": [[["conv2d_26", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_25", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "activation": "relu"}, "name": "activation_25", "inbound_nodes": [[["batch_normalization_25", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_5", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_5", "inbound_nodes": [[["activation_25", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_5", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": -1}, "name": "concatenate_5", "inbound_nodes": [[["conv2d_transpose_5", 0, 0, {}], ["activation_15", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_11", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_11", "inbound_nodes": [[["concatenate_5", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_28", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_28", "inbound_nodes": [[["dropout_11", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_27", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_27", "inbound_nodes": [[["conv2d_28", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_27", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "activation": "relu"}, "name": "activation_27", "inbound_nodes": [[["batch_normalization_27", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_29", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 7, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_29", "inbound_nodes": [[["activation_27", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["conv2d_29", 0, 0]]}}, "training_config": {"loss": "mse", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "iou_coef", "dtype": "float32", "fn": "iou_coef"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "LossScaleOptimizer", "config": {"inner_optimizer": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}, "dynamic": true, "initial_scale": 32768.0, "dynamic_growth_steps": 2000}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 544, 736, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 544, 736, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
?


-kernel
.bias
/regularization_losses
0	variables
1trainable_variables
2	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_16", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_16", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 544, 736, 3]}}
?

3axis
	4gamma
5beta
6moving_mean
7moving_variance
8regularization_losses
9	variables
:trainable_variables
;	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_15", "trainable": true, "expects_training_arg": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_15", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 544, 736, 32]}}
?
<regularization_losses
=	variables
>trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_15", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_15", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "activation": "relu"}}
?
@regularization_losses
A	variables
Btrainable_variables
C	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_3", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
Dregularization_losses
E	variables
Ftrainable_variables
G	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_6", "trainable": true, "expects_training_arg": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_6", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "rate": 0.1, "noise_shape": null, "seed": null}}
?


Hkernel
Ibias
Jregularization_losses
K	variables
Ltrainable_variables
M	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_18", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_18", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 272, 368, 32]}}
?

Naxis
	Ogamma
Pbeta
Qmoving_mean
Rmoving_variance
Sregularization_losses
T	variables
Utrainable_variables
V	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_17", "trainable": true, "expects_training_arg": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_17", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 272, 368, 64]}}
?
Wregularization_losses
X	variables
Ytrainable_variables
Z	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_17", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_17", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "activation": "relu"}}
?
[regularization_losses
\	variables
]trainable_variables
^	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_4", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
_regularization_losses
`	variables
atrainable_variables
b	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_7", "trainable": true, "expects_training_arg": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_7", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "rate": 0.1, "noise_shape": null, "seed": null}}
?


ckernel
dbias
eregularization_losses
f	variables
gtrainable_variables
h	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_20", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_20", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 136, 184, 64]}}
?

iaxis
	jgamma
kbeta
lmoving_mean
mmoving_variance
nregularization_losses
o	variables
ptrainable_variables
q	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_19", "trainable": true, "expects_training_arg": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_19", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 136, 184, 128]}}
?
rregularization_losses
s	variables
ttrainable_variables
u	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_19", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_19", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "activation": "relu"}}
?
vregularization_losses
w	variables
xtrainable_variables
y	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_5", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_5", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
zregularization_losses
{	variables
|trainable_variables
}	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_8", "trainable": true, "expects_training_arg": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_8", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "rate": 0.1, "noise_shape": null, "seed": null}}
?


~kernel
bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_22", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_22", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 68, 92, 128]}}
?

	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_21", "trainable": true, "expects_training_arg": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_21", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 68, 92, 256]}}
?
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_21", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_21", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "activation": "relu"}}
?
?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_3", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose_3", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 68, 92, 256]}}
?
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate_3", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate_3", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 136, 184, 128]}, {"class_name": "TensorShape", "items": [null, 136, 184, 128]}]}
?
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_9", "trainable": true, "expects_training_arg": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_9", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "rate": 0.1, "noise_shape": null, "seed": null}}
?

?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_24", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_24", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 136, 184, 256]}}
?

	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_23", "trainable": true, "expects_training_arg": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_23", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 136, 184, 128]}}
?
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_23", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_23", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "activation": "relu"}}
?
?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_4", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose_4", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 136, 184, 128]}}
?
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate_4", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate_4", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 272, 368, 64]}, {"class_name": "TensorShape", "items": [null, 272, 368, 64]}]}
?
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_10", "trainable": true, "expects_training_arg": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_10", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "rate": 0.1, "noise_shape": null, "seed": null}}
?

?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_26", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_26", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 272, 368, 128]}}
?

	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_25", "trainable": true, "expects_training_arg": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_25", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 272, 368, 64]}}
?
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_25", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_25", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "activation": "relu"}}
?
?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_5", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose_5", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 272, 368, 64]}}
?
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate_5", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate_5", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 544, 736, 32]}, {"class_name": "TensorShape", "items": [null, 544, 736, 32]}]}
?
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_11", "trainable": true, "expects_training_arg": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_11", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "rate": 0.1, "noise_shape": null, "seed": null}}
?

?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_28", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_28", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 544, 736, 64]}}
?

	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_27", "trainable": true, "expects_training_arg": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_27", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 544, 736, 32]}}
?
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_27", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_27", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "activation": "relu"}}
?

?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_29", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_29", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 7, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 544, 736, 32]}}
?
?
loss_scale
?base_optimizer
	?iter
?beta_1
?beta_2

?decay
?learning_rate-m?.m?4m?5m?Hm?Im?Om?Pm?cm?dm?jm?km?~m?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?-v?.v?4v?5v?Hv?Iv?Ov?Pv?cv?dv?jv?kv?~v?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?"
	optimizer
?
-0
.1
42
53
64
75
H6
I7
O8
P9
Q10
R11
c12
d13
j14
k15
l16
m17
~18
19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45
?46
?47
?48
?49"
trackable_list_wrapper
?
-0
.1
42
53
H4
I5
O6
P7
c8
d9
j10
k11
~12
13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
(	variables
)trainable_variables
*regularization_losses
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
?metrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
*:( 2conv2d_16/kernel
: 2conv2d_16/bias
 "
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
?
/regularization_losses
0	variables
1trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:( 2batch_normalization_15/gamma
):' 2batch_normalization_15/beta
2:0  (2"batch_normalization_15/moving_mean
6:4  (2&batch_normalization_15/moving_variance
 "
trackable_list_wrapper
<
40
51
62
73"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
?
8regularization_losses
9	variables
:trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
<regularization_losses
=	variables
>trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
@regularization_losses
A	variables
Btrainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Dregularization_losses
E	variables
Ftrainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:( @2conv2d_18/kernel
:@2conv2d_18/bias
 "
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
?
Jregularization_losses
K	variables
Ltrainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2batch_normalization_17/gamma
):'@2batch_normalization_17/beta
2:0@ (2"batch_normalization_17/moving_mean
6:4@ (2&batch_normalization_17/moving_variance
 "
trackable_list_wrapper
<
O0
P1
Q2
R3"
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
?
Sregularization_losses
T	variables
Utrainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Wregularization_losses
X	variables
Ytrainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
[regularization_losses
\	variables
]trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
_regularization_losses
`	variables
atrainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)@?2conv2d_20/kernel
:?2conv2d_20/bias
 "
trackable_list_wrapper
.
c0
d1"
trackable_list_wrapper
.
c0
d1"
trackable_list_wrapper
?
eregularization_losses
f	variables
gtrainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)?2batch_normalization_19/gamma
*:(?2batch_normalization_19/beta
3:1? (2"batch_normalization_19/moving_mean
7:5? (2&batch_normalization_19/moving_variance
 "
trackable_list_wrapper
<
j0
k1
l2
m3"
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
?
nregularization_losses
o	variables
ptrainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
rregularization_losses
s	variables
ttrainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
vregularization_losses
w	variables
xtrainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
zregularization_losses
{	variables
|trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
,:*??2conv2d_22/kernel
:?2conv2d_22/bias
 "
trackable_list_wrapper
.
~0
1"
trackable_list_wrapper
.
~0
1"
trackable_list_wrapper
?
?regularization_losses
?	variables
?trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)?2batch_normalization_21/gamma
*:(?2batch_normalization_21/beta
3:1? (2"batch_normalization_21/moving_mean
7:5? (2&batch_normalization_21/moving_variance
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?regularization_losses
?	variables
?trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?	variables
?trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
5:3??2conv2d_transpose_3/kernel
&:$?2conv2d_transpose_3/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?regularization_losses
?	variables
?trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?	variables
?trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?	variables
?trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
,:*??2conv2d_24/kernel
:?2conv2d_24/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?regularization_losses
?	variables
?trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)?2batch_normalization_23/gamma
*:(?2batch_normalization_23/beta
3:1? (2"batch_normalization_23/moving_mean
7:5? (2&batch_normalization_23/moving_variance
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?regularization_losses
?	variables
?trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?	variables
?trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
4:2@?2conv2d_transpose_4/kernel
%:#@2conv2d_transpose_4/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?regularization_losses
?	variables
?trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?	variables
?trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?	variables
?trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)?@2conv2d_26/kernel
:@2conv2d_26/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?regularization_losses
?	variables
?trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2batch_normalization_25/gamma
):'@2batch_normalization_25/beta
2:0@ (2"batch_normalization_25/moving_mean
6:4@ (2&batch_normalization_25/moving_variance
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?regularization_losses
?	variables
?trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?	variables
?trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
3:1 @2conv2d_transpose_5/kernel
%:# 2conv2d_transpose_5/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?regularization_losses
?	variables
?trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?	variables
?trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?	variables
?trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(@ 2conv2d_28/kernel
: 2conv2d_28/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?regularization_losses
?	variables
?trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:( 2batch_normalization_27/gamma
):' 2batch_normalization_27/beta
2:0  (2"batch_normalization_27/moving_mean
6:4  (2&batch_normalization_27/moving_variance
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?regularization_losses
?	variables
?trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?	variables
?trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:( 2conv2d_29/kernel
:2conv2d_29/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?regularization_losses
?	variables
?trainable_variables
?metrics
?layers
 ?layer_regularization_losses
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
H
?current_loss_scale
?
good_steps"
_generic_user_object
"
_generic_user_object
:	 (2cond_1/Adam/iter
: (2cond_1/Adam/beta_1
: (2cond_1/Adam/beta_2
: (2cond_1/Adam/decay
#:! (2cond_1/Adam/learning_rate
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?
60
71
Q2
R3
l4
m5
?6
?7
?8
?9
?10
?11
?12
?13"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
l0
m1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
: 2current_loss_scale
:	 2
good_steps
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "iou_coef", "dtype": "float32", "config": {"name": "iou_coef", "dtype": "float32", "fn": "iou_coef"}}
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
6:4 2cond_1/Adam/conv2d_16/kernel/m
(:& 2cond_1/Adam/conv2d_16/bias/m
6:4 2*cond_1/Adam/batch_normalization_15/gamma/m
5:3 2)cond_1/Adam/batch_normalization_15/beta/m
6:4 @2cond_1/Adam/conv2d_18/kernel/m
(:&@2cond_1/Adam/conv2d_18/bias/m
6:4@2*cond_1/Adam/batch_normalization_17/gamma/m
5:3@2)cond_1/Adam/batch_normalization_17/beta/m
7:5@?2cond_1/Adam/conv2d_20/kernel/m
):'?2cond_1/Adam/conv2d_20/bias/m
7:5?2*cond_1/Adam/batch_normalization_19/gamma/m
6:4?2)cond_1/Adam/batch_normalization_19/beta/m
8:6??2cond_1/Adam/conv2d_22/kernel/m
):'?2cond_1/Adam/conv2d_22/bias/m
7:5?2*cond_1/Adam/batch_normalization_21/gamma/m
6:4?2)cond_1/Adam/batch_normalization_21/beta/m
A:???2'cond_1/Adam/conv2d_transpose_3/kernel/m
2:0?2%cond_1/Adam/conv2d_transpose_3/bias/m
8:6??2cond_1/Adam/conv2d_24/kernel/m
):'?2cond_1/Adam/conv2d_24/bias/m
7:5?2*cond_1/Adam/batch_normalization_23/gamma/m
6:4?2)cond_1/Adam/batch_normalization_23/beta/m
@:>@?2'cond_1/Adam/conv2d_transpose_4/kernel/m
1:/@2%cond_1/Adam/conv2d_transpose_4/bias/m
7:5?@2cond_1/Adam/conv2d_26/kernel/m
(:&@2cond_1/Adam/conv2d_26/bias/m
6:4@2*cond_1/Adam/batch_normalization_25/gamma/m
5:3@2)cond_1/Adam/batch_normalization_25/beta/m
?:= @2'cond_1/Adam/conv2d_transpose_5/kernel/m
1:/ 2%cond_1/Adam/conv2d_transpose_5/bias/m
6:4@ 2cond_1/Adam/conv2d_28/kernel/m
(:& 2cond_1/Adam/conv2d_28/bias/m
6:4 2*cond_1/Adam/batch_normalization_27/gamma/m
5:3 2)cond_1/Adam/batch_normalization_27/beta/m
6:4 2cond_1/Adam/conv2d_29/kernel/m
(:&2cond_1/Adam/conv2d_29/bias/m
6:4 2cond_1/Adam/conv2d_16/kernel/v
(:& 2cond_1/Adam/conv2d_16/bias/v
6:4 2*cond_1/Adam/batch_normalization_15/gamma/v
5:3 2)cond_1/Adam/batch_normalization_15/beta/v
6:4 @2cond_1/Adam/conv2d_18/kernel/v
(:&@2cond_1/Adam/conv2d_18/bias/v
6:4@2*cond_1/Adam/batch_normalization_17/gamma/v
5:3@2)cond_1/Adam/batch_normalization_17/beta/v
7:5@?2cond_1/Adam/conv2d_20/kernel/v
):'?2cond_1/Adam/conv2d_20/bias/v
7:5?2*cond_1/Adam/batch_normalization_19/gamma/v
6:4?2)cond_1/Adam/batch_normalization_19/beta/v
8:6??2cond_1/Adam/conv2d_22/kernel/v
):'?2cond_1/Adam/conv2d_22/bias/v
7:5?2*cond_1/Adam/batch_normalization_21/gamma/v
6:4?2)cond_1/Adam/batch_normalization_21/beta/v
A:???2'cond_1/Adam/conv2d_transpose_3/kernel/v
2:0?2%cond_1/Adam/conv2d_transpose_3/bias/v
8:6??2cond_1/Adam/conv2d_24/kernel/v
):'?2cond_1/Adam/conv2d_24/bias/v
7:5?2*cond_1/Adam/batch_normalization_23/gamma/v
6:4?2)cond_1/Adam/batch_normalization_23/beta/v
@:>@?2'cond_1/Adam/conv2d_transpose_4/kernel/v
1:/@2%cond_1/Adam/conv2d_transpose_4/bias/v
7:5?@2cond_1/Adam/conv2d_26/kernel/v
(:&@2cond_1/Adam/conv2d_26/bias/v
6:4@2*cond_1/Adam/batch_normalization_25/gamma/v
5:3@2)cond_1/Adam/batch_normalization_25/beta/v
?:= @2'cond_1/Adam/conv2d_transpose_5/kernel/v
1:/ 2%cond_1/Adam/conv2d_transpose_5/bias/v
6:4@ 2cond_1/Adam/conv2d_28/kernel/v
(:& 2cond_1/Adam/conv2d_28/bias/v
6:4 2*cond_1/Adam/batch_normalization_27/gamma/v
5:3 2)cond_1/Adam/batch_normalization_27/beta/v
6:4 2cond_1/Adam/conv2d_29/kernel/v
(:&2cond_1/Adam/conv2d_29/bias/v
?2?
B__inference_model_1_layer_call_and_return_conditional_losses_40572
B__inference_model_1_layer_call_and_return_conditional_losses_39653
B__inference_model_1_layer_call_and_return_conditional_losses_40825
B__inference_model_1_layer_call_and_return_conditional_losses_39511?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
 __inference__wrapped_model_37530?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *0?-
+?(
input_2???????????
?2?
'__inference_model_1_layer_call_fn_39901
'__inference_model_1_layer_call_fn_40930
'__inference_model_1_layer_call_fn_41035
'__inference_model_1_layer_call_fn_40148?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_conv2d_16_layer_call_and_return_conditional_losses_41047?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_16_layer_call_fn_41056?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_41140
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_41094
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_41158
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_41076?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
6__inference_batch_normalization_15_layer_call_fn_41184
6__inference_batch_normalization_15_layer_call_fn_41120
6__inference_batch_normalization_15_layer_call_fn_41171
6__inference_batch_normalization_15_layer_call_fn_41107?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_activation_15_layer_call_and_return_conditional_losses_41189?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_activation_15_layer_call_fn_41194?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_37640?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
/__inference_max_pooling2d_3_layer_call_fn_37646?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
D__inference_dropout_6_layer_call_and_return_conditional_losses_41211
D__inference_dropout_6_layer_call_and_return_conditional_losses_41206?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dropout_6_layer_call_fn_41216
)__inference_dropout_6_layer_call_fn_41221?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_conv2d_18_layer_call_and_return_conditional_losses_41233?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_18_layer_call_fn_41242?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_41344
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_41326
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_41280
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_41262?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
6__inference_batch_normalization_17_layer_call_fn_41357
6__inference_batch_normalization_17_layer_call_fn_41370
6__inference_batch_normalization_17_layer_call_fn_41293
6__inference_batch_normalization_17_layer_call_fn_41306?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_activation_17_layer_call_and_return_conditional_losses_41375?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_activation_17_layer_call_fn_41380?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_37756?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
/__inference_max_pooling2d_4_layer_call_fn_37762?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
D__inference_dropout_7_layer_call_and_return_conditional_losses_41392
D__inference_dropout_7_layer_call_and_return_conditional_losses_41397?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dropout_7_layer_call_fn_41402
)__inference_dropout_7_layer_call_fn_41407?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_conv2d_20_layer_call_and_return_conditional_losses_41419?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_20_layer_call_fn_41428?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_41448
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_41530
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_41466
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_41512?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
6__inference_batch_normalization_19_layer_call_fn_41492
6__inference_batch_normalization_19_layer_call_fn_41479
6__inference_batch_normalization_19_layer_call_fn_41556
6__inference_batch_normalization_19_layer_call_fn_41543?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_activation_19_layer_call_and_return_conditional_losses_41561?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_activation_19_layer_call_fn_41566?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_37872?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
/__inference_max_pooling2d_5_layer_call_fn_37878?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
D__inference_dropout_8_layer_call_and_return_conditional_losses_41578
D__inference_dropout_8_layer_call_and_return_conditional_losses_41583?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dropout_8_layer_call_fn_41593
)__inference_dropout_8_layer_call_fn_41588?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_conv2d_22_layer_call_and_return_conditional_losses_41605?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_22_layer_call_fn_41614?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
Q__inference_batch_normalization_21_layer_call_and_return_conditional_losses_41698
Q__inference_batch_normalization_21_layer_call_and_return_conditional_losses_41652
Q__inference_batch_normalization_21_layer_call_and_return_conditional_losses_41716
Q__inference_batch_normalization_21_layer_call_and_return_conditional_losses_41634?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
6__inference_batch_normalization_21_layer_call_fn_41678
6__inference_batch_normalization_21_layer_call_fn_41742
6__inference_batch_normalization_21_layer_call_fn_41729
6__inference_batch_normalization_21_layer_call_fn_41665?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_activation_21_layer_call_and_return_conditional_losses_41747?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_activation_21_layer_call_fn_41752?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_38018?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0,????????????????????????????
?2?
2__inference_conv2d_transpose_3_layer_call_fn_38028?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0,????????????????????????????
?2?
H__inference_concatenate_3_layer_call_and_return_conditional_losses_41759?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_concatenate_3_layer_call_fn_41765?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dropout_9_layer_call_and_return_conditional_losses_41777
D__inference_dropout_9_layer_call_and_return_conditional_losses_41782?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dropout_9_layer_call_fn_41792
)__inference_dropout_9_layer_call_fn_41787?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_conv2d_24_layer_call_and_return_conditional_losses_41804?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_24_layer_call_fn_41813?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
Q__inference_batch_normalization_23_layer_call_and_return_conditional_losses_41851
Q__inference_batch_normalization_23_layer_call_and_return_conditional_losses_41897
Q__inference_batch_normalization_23_layer_call_and_return_conditional_losses_41833
Q__inference_batch_normalization_23_layer_call_and_return_conditional_losses_41915?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
6__inference_batch_normalization_23_layer_call_fn_41941
6__inference_batch_normalization_23_layer_call_fn_41928
6__inference_batch_normalization_23_layer_call_fn_41877
6__inference_batch_normalization_23_layer_call_fn_41864?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_activation_23_layer_call_and_return_conditional_losses_41946?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_activation_23_layer_call_fn_41951?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
M__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_38168?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0,????????????????????????????
?2?
2__inference_conv2d_transpose_4_layer_call_fn_38178?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0,????????????????????????????
?2?
H__inference_concatenate_4_layer_call_and_return_conditional_losses_41958?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_concatenate_4_layer_call_fn_41964?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dropout_10_layer_call_and_return_conditional_losses_41976
E__inference_dropout_10_layer_call_and_return_conditional_losses_41981?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_dropout_10_layer_call_fn_41991
*__inference_dropout_10_layer_call_fn_41986?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_conv2d_26_layer_call_and_return_conditional_losses_42003?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_26_layer_call_fn_42012?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
Q__inference_batch_normalization_25_layer_call_and_return_conditional_losses_42050
Q__inference_batch_normalization_25_layer_call_and_return_conditional_losses_42032
Q__inference_batch_normalization_25_layer_call_and_return_conditional_losses_42096
Q__inference_batch_normalization_25_layer_call_and_return_conditional_losses_42114?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
6__inference_batch_normalization_25_layer_call_fn_42076
6__inference_batch_normalization_25_layer_call_fn_42140
6__inference_batch_normalization_25_layer_call_fn_42127
6__inference_batch_normalization_25_layer_call_fn_42063?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_activation_25_layer_call_and_return_conditional_losses_42145?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_activation_25_layer_call_fn_42150?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
M__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_38318?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????@
?2?
2__inference_conv2d_transpose_5_layer_call_fn_38328?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????@
?2?
H__inference_concatenate_5_layer_call_and_return_conditional_losses_42157?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_concatenate_5_layer_call_fn_42163?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dropout_11_layer_call_and_return_conditional_losses_42180
E__inference_dropout_11_layer_call_and_return_conditional_losses_42175?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_dropout_11_layer_call_fn_42185
*__inference_dropout_11_layer_call_fn_42190?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_conv2d_28_layer_call_and_return_conditional_losses_42202?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_28_layer_call_fn_42211?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
Q__inference_batch_normalization_27_layer_call_and_return_conditional_losses_42295
Q__inference_batch_normalization_27_layer_call_and_return_conditional_losses_42313
Q__inference_batch_normalization_27_layer_call_and_return_conditional_losses_42231
Q__inference_batch_normalization_27_layer_call_and_return_conditional_losses_42249?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
6__inference_batch_normalization_27_layer_call_fn_42339
6__inference_batch_normalization_27_layer_call_fn_42262
6__inference_batch_normalization_27_layer_call_fn_42275
6__inference_batch_normalization_27_layer_call_fn_42326?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_activation_27_layer_call_and_return_conditional_losses_42344?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_activation_27_layer_call_fn_42349?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_29_layer_call_and_return_conditional_losses_42362?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_29_layer_call_fn_42371?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_40263input_2"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
 __inference__wrapped_model_37530?P-.4567HIOPQRcdjklm~??????????????????????????????:?7
0?-
+?(
input_2???????????
? "??<
:
	conv2d_29-?*
	conv2d_29????????????
H__inference_activation_15_layer_call_and_return_conditional_losses_41189l9?6
/?,
*?'
inputs??????????? 
? "/?,
%?"
0??????????? 
? ?
-__inference_activation_15_layer_call_fn_41194_9?6
/?,
*?'
inputs??????????? 
? ""???????????? ?
H__inference_activation_17_layer_call_and_return_conditional_losses_41375l9?6
/?,
*?'
inputs???????????@
? "/?,
%?"
0???????????@
? ?
-__inference_activation_17_layer_call_fn_41380_9?6
/?,
*?'
inputs???????????@
? ""????????????@?
H__inference_activation_19_layer_call_and_return_conditional_losses_41561n:?7
0?-
+?(
inputs????????????
? "0?-
&?#
0????????????
? ?
-__inference_activation_19_layer_call_fn_41566a:?7
0?-
+?(
inputs????????????
? "#? ?????????????
H__inference_activation_21_layer_call_and_return_conditional_losses_41747j8?5
.?+
)?&
inputs?????????D\?
? ".?+
$?!
0?????????D\?
? ?
-__inference_activation_21_layer_call_fn_41752]8?5
.?+
)?&
inputs?????????D\?
? "!??????????D\??
H__inference_activation_23_layer_call_and_return_conditional_losses_41946n:?7
0?-
+?(
inputs????????????
? "0?-
&?#
0????????????
? ?
-__inference_activation_23_layer_call_fn_41951a:?7
0?-
+?(
inputs????????????
? "#? ?????????????
H__inference_activation_25_layer_call_and_return_conditional_losses_42145l9?6
/?,
*?'
inputs???????????@
? "/?,
%?"
0???????????@
? ?
-__inference_activation_25_layer_call_fn_42150_9?6
/?,
*?'
inputs???????????@
? ""????????????@?
H__inference_activation_27_layer_call_and_return_conditional_losses_42344l9?6
/?,
*?'
inputs??????????? 
? "/?,
%?"
0??????????? 
? ?
-__inference_activation_27_layer_call_fn_42349_9?6
/?,
*?'
inputs??????????? 
? ""???????????? ?
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_41076?4567M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_41094?4567M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_41140v4567=?:
3?0
*?'
inputs??????????? 
p
? "/?,
%?"
0??????????? 
? ?
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_41158v4567=?:
3?0
*?'
inputs??????????? 
p 
? "/?,
%?"
0??????????? 
? ?
6__inference_batch_normalization_15_layer_call_fn_41107?4567M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
6__inference_batch_normalization_15_layer_call_fn_41120?4567M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
6__inference_batch_normalization_15_layer_call_fn_41171i4567=?:
3?0
*?'
inputs??????????? 
p
? ""???????????? ?
6__inference_batch_normalization_15_layer_call_fn_41184i4567=?:
3?0
*?'
inputs??????????? 
p 
? ""???????????? ?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_41262?OPQRM?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_41280?OPQRM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_41326vOPQR=?:
3?0
*?'
inputs???????????@
p
? "/?,
%?"
0???????????@
? ?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_41344vOPQR=?:
3?0
*?'
inputs???????????@
p 
? "/?,
%?"
0???????????@
? ?
6__inference_batch_normalization_17_layer_call_fn_41293?OPQRM?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
6__inference_batch_normalization_17_layer_call_fn_41306?OPQRM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
6__inference_batch_normalization_17_layer_call_fn_41357iOPQR=?:
3?0
*?'
inputs???????????@
p
? ""????????????@?
6__inference_batch_normalization_17_layer_call_fn_41370iOPQR=?:
3?0
*?'
inputs???????????@
p 
? ""????????????@?
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_41448?jklmN?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_41466?jklmN?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_41512xjklm>?;
4?1
+?(
inputs????????????
p
? "0?-
&?#
0????????????
? ?
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_41530xjklm>?;
4?1
+?(
inputs????????????
p 
? "0?-
&?#
0????????????
? ?
6__inference_batch_normalization_19_layer_call_fn_41479?jklmN?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
6__inference_batch_normalization_19_layer_call_fn_41492?jklmN?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
6__inference_batch_normalization_19_layer_call_fn_41543kjklm>?;
4?1
+?(
inputs????????????
p
? "#? ?????????????
6__inference_batch_normalization_19_layer_call_fn_41556kjklm>?;
4?1
+?(
inputs????????????
p 
? "#? ?????????????
Q__inference_batch_normalization_21_layer_call_and_return_conditional_losses_41634?????N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
Q__inference_batch_normalization_21_layer_call_and_return_conditional_losses_41652?????N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
Q__inference_batch_normalization_21_layer_call_and_return_conditional_losses_41698x????<?9
2?/
)?&
inputs?????????D\?
p
? ".?+
$?!
0?????????D\?
? ?
Q__inference_batch_normalization_21_layer_call_and_return_conditional_losses_41716x????<?9
2?/
)?&
inputs?????????D\?
p 
? ".?+
$?!
0?????????D\?
? ?
6__inference_batch_normalization_21_layer_call_fn_41665?????N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
6__inference_batch_normalization_21_layer_call_fn_41678?????N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
6__inference_batch_normalization_21_layer_call_fn_41729k????<?9
2?/
)?&
inputs?????????D\?
p
? "!??????????D\??
6__inference_batch_normalization_21_layer_call_fn_41742k????<?9
2?/
)?&
inputs?????????D\?
p 
? "!??????????D\??
Q__inference_batch_normalization_23_layer_call_and_return_conditional_losses_41833?????N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
Q__inference_batch_normalization_23_layer_call_and_return_conditional_losses_41851?????N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
Q__inference_batch_normalization_23_layer_call_and_return_conditional_losses_41897|????>?;
4?1
+?(
inputs????????????
p
? "0?-
&?#
0????????????
? ?
Q__inference_batch_normalization_23_layer_call_and_return_conditional_losses_41915|????>?;
4?1
+?(
inputs????????????
p 
? "0?-
&?#
0????????????
? ?
6__inference_batch_normalization_23_layer_call_fn_41864?????N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
6__inference_batch_normalization_23_layer_call_fn_41877?????N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
6__inference_batch_normalization_23_layer_call_fn_41928o????>?;
4?1
+?(
inputs????????????
p
? "#? ?????????????
6__inference_batch_normalization_23_layer_call_fn_41941o????>?;
4?1
+?(
inputs????????????
p 
? "#? ?????????????
Q__inference_batch_normalization_25_layer_call_and_return_conditional_losses_42032?????M?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
Q__inference_batch_normalization_25_layer_call_and_return_conditional_losses_42050?????M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
Q__inference_batch_normalization_25_layer_call_and_return_conditional_losses_42096z????=?:
3?0
*?'
inputs???????????@
p
? "/?,
%?"
0???????????@
? ?
Q__inference_batch_normalization_25_layer_call_and_return_conditional_losses_42114z????=?:
3?0
*?'
inputs???????????@
p 
? "/?,
%?"
0???????????@
? ?
6__inference_batch_normalization_25_layer_call_fn_42063?????M?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
6__inference_batch_normalization_25_layer_call_fn_42076?????M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
6__inference_batch_normalization_25_layer_call_fn_42127m????=?:
3?0
*?'
inputs???????????@
p
? ""????????????@?
6__inference_batch_normalization_25_layer_call_fn_42140m????=?:
3?0
*?'
inputs???????????@
p 
? ""????????????@?
Q__inference_batch_normalization_27_layer_call_and_return_conditional_losses_42231z????=?:
3?0
*?'
inputs??????????? 
p
? "/?,
%?"
0??????????? 
? ?
Q__inference_batch_normalization_27_layer_call_and_return_conditional_losses_42249z????=?:
3?0
*?'
inputs??????????? 
p 
? "/?,
%?"
0??????????? 
? ?
Q__inference_batch_normalization_27_layer_call_and_return_conditional_losses_42295?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
Q__inference_batch_normalization_27_layer_call_and_return_conditional_losses_42313?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
6__inference_batch_normalization_27_layer_call_fn_42262m????=?:
3?0
*?'
inputs??????????? 
p
? ""???????????? ?
6__inference_batch_normalization_27_layer_call_fn_42275m????=?:
3?0
*?'
inputs??????????? 
p 
? ""???????????? ?
6__inference_batch_normalization_27_layer_call_fn_42326?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
6__inference_batch_normalization_27_layer_call_fn_42339?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
H__inference_concatenate_3_layer_call_and_return_conditional_losses_41759???}
v?s
q?n
=?:
inputs/0,????????????????????????????
-?*
inputs/1????????????
? "0?-
&?#
0????????????
? ?
-__inference_concatenate_3_layer_call_fn_41765???}
v?s
q?n
=?:
inputs/0,????????????????????????????
-?*
inputs/1????????????
? "#? ?????????????
H__inference_concatenate_4_layer_call_and_return_conditional_losses_41958?~?{
t?q
o?l
<?9
inputs/0+???????????????????????????@
,?)
inputs/1???????????@
? "0?-
&?#
0????????????
? ?
-__inference_concatenate_4_layer_call_fn_41964?~?{
t?q
o?l
<?9
inputs/0+???????????????????????????@
,?)
inputs/1???????????@
? "#? ?????????????
H__inference_concatenate_5_layer_call_and_return_conditional_losses_42157?~?{
t?q
o?l
<?9
inputs/0+??????????????????????????? 
,?)
inputs/1??????????? 
? "/?,
%?"
0???????????@
? ?
-__inference_concatenate_5_layer_call_fn_42163?~?{
t?q
o?l
<?9
inputs/0+??????????????????????????? 
,?)
inputs/1??????????? 
? ""????????????@?
D__inference_conv2d_16_layer_call_and_return_conditional_losses_41047p-.9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0??????????? 
? ?
)__inference_conv2d_16_layer_call_fn_41056c-.9?6
/?,
*?'
inputs???????????
? ""???????????? ?
D__inference_conv2d_18_layer_call_and_return_conditional_losses_41233pHI9?6
/?,
*?'
inputs??????????? 
? "/?,
%?"
0???????????@
? ?
)__inference_conv2d_18_layer_call_fn_41242cHI9?6
/?,
*?'
inputs??????????? 
? ""????????????@?
D__inference_conv2d_20_layer_call_and_return_conditional_losses_41419qcd9?6
/?,
*?'
inputs???????????@
? "0?-
&?#
0????????????
? ?
)__inference_conv2d_20_layer_call_fn_41428dcd9?6
/?,
*?'
inputs???????????@
? "#? ?????????????
D__inference_conv2d_22_layer_call_and_return_conditional_losses_41605n~8?5
.?+
)?&
inputs?????????D\?
? ".?+
$?!
0?????????D\?
? ?
)__inference_conv2d_22_layer_call_fn_41614a~8?5
.?+
)?&
inputs?????????D\?
? "!??????????D\??
D__inference_conv2d_24_layer_call_and_return_conditional_losses_41804t??:?7
0?-
+?(
inputs????????????
? "0?-
&?#
0????????????
? ?
)__inference_conv2d_24_layer_call_fn_41813g??:?7
0?-
+?(
inputs????????????
? "#? ?????????????
D__inference_conv2d_26_layer_call_and_return_conditional_losses_42003s??:?7
0?-
+?(
inputs????????????
? "/?,
%?"
0???????????@
? ?
)__inference_conv2d_26_layer_call_fn_42012f??:?7
0?-
+?(
inputs????????????
? ""????????????@?
D__inference_conv2d_28_layer_call_and_return_conditional_losses_42202r??9?6
/?,
*?'
inputs???????????@
? "/?,
%?"
0??????????? 
? ?
)__inference_conv2d_28_layer_call_fn_42211e??9?6
/?,
*?'
inputs???????????@
? ""???????????? ?
D__inference_conv2d_29_layer_call_and_return_conditional_losses_42362r??9?6
/?,
*?'
inputs??????????? 
? "/?,
%?"
0???????????
? ?
)__inference_conv2d_29_layer_call_fn_42371e??9?6
/?,
*?'
inputs??????????? 
? ""?????????????
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_38018???J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
2__inference_conv2d_transpose_3_layer_call_fn_38028???J?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
M__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_38168???J?G
@?=
;?8
inputs,????????????????????????????
? "??<
5?2
0+???????????????????????????@
? ?
2__inference_conv2d_transpose_4_layer_call_fn_38178???J?G
@?=
;?8
inputs,????????????????????????????
? "2?/+???????????????????????????@?
M__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_38318???I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+??????????????????????????? 
? ?
2__inference_conv2d_transpose_5_layer_call_fn_38328???I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+??????????????????????????? ?
E__inference_dropout_10_layer_call_and_return_conditional_losses_41976r>?;
4?1
+?(
inputs????????????
p
? "0?-
&?#
0????????????
? ?
E__inference_dropout_10_layer_call_and_return_conditional_losses_41981r>?;
4?1
+?(
inputs????????????
p 
? "0?-
&?#
0????????????
? ?
*__inference_dropout_10_layer_call_fn_41986e>?;
4?1
+?(
inputs????????????
p
? "#? ?????????????
*__inference_dropout_10_layer_call_fn_41991e>?;
4?1
+?(
inputs????????????
p 
? "#? ?????????????
E__inference_dropout_11_layer_call_and_return_conditional_losses_42175p=?:
3?0
*?'
inputs???????????@
p
? "/?,
%?"
0???????????@
? ?
E__inference_dropout_11_layer_call_and_return_conditional_losses_42180p=?:
3?0
*?'
inputs???????????@
p 
? "/?,
%?"
0???????????@
? ?
*__inference_dropout_11_layer_call_fn_42185c=?:
3?0
*?'
inputs???????????@
p
? ""????????????@?
*__inference_dropout_11_layer_call_fn_42190c=?:
3?0
*?'
inputs???????????@
p 
? ""????????????@?
D__inference_dropout_6_layer_call_and_return_conditional_losses_41206p=?:
3?0
*?'
inputs??????????? 
p
? "/?,
%?"
0??????????? 
? ?
D__inference_dropout_6_layer_call_and_return_conditional_losses_41211p=?:
3?0
*?'
inputs??????????? 
p 
? "/?,
%?"
0??????????? 
? ?
)__inference_dropout_6_layer_call_fn_41216c=?:
3?0
*?'
inputs??????????? 
p
? ""???????????? ?
)__inference_dropout_6_layer_call_fn_41221c=?:
3?0
*?'
inputs??????????? 
p 
? ""???????????? ?
D__inference_dropout_7_layer_call_and_return_conditional_losses_41392p=?:
3?0
*?'
inputs???????????@
p
? "/?,
%?"
0???????????@
? ?
D__inference_dropout_7_layer_call_and_return_conditional_losses_41397p=?:
3?0
*?'
inputs???????????@
p 
? "/?,
%?"
0???????????@
? ?
)__inference_dropout_7_layer_call_fn_41402c=?:
3?0
*?'
inputs???????????@
p
? ""????????????@?
)__inference_dropout_7_layer_call_fn_41407c=?:
3?0
*?'
inputs???????????@
p 
? ""????????????@?
D__inference_dropout_8_layer_call_and_return_conditional_losses_41578n<?9
2?/
)?&
inputs?????????D\?
p
? ".?+
$?!
0?????????D\?
? ?
D__inference_dropout_8_layer_call_and_return_conditional_losses_41583n<?9
2?/
)?&
inputs?????????D\?
p 
? ".?+
$?!
0?????????D\?
? ?
)__inference_dropout_8_layer_call_fn_41588a<?9
2?/
)?&
inputs?????????D\?
p
? "!??????????D\??
)__inference_dropout_8_layer_call_fn_41593a<?9
2?/
)?&
inputs?????????D\?
p 
? "!??????????D\??
D__inference_dropout_9_layer_call_and_return_conditional_losses_41777r>?;
4?1
+?(
inputs????????????
p
? "0?-
&?#
0????????????
? ?
D__inference_dropout_9_layer_call_and_return_conditional_losses_41782r>?;
4?1
+?(
inputs????????????
p 
? "0?-
&?#
0????????????
? ?
)__inference_dropout_9_layer_call_fn_41787e>?;
4?1
+?(
inputs????????????
p
? "#? ?????????????
)__inference_dropout_9_layer_call_fn_41792e>?;
4?1
+?(
inputs????????????
p 
? "#? ?????????????
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_37640?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_max_pooling2d_3_layer_call_fn_37646?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_37756?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_max_pooling2d_4_layer_call_fn_37762?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_37872?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_max_pooling2d_5_layer_call_fn_37878?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
B__inference_model_1_layer_call_and_return_conditional_losses_39511?P-.4567HIOPQRcdjklm~??????????????????????????????B??
8?5
+?(
input_2???????????
p

 
? "/?,
%?"
0???????????
? ?
B__inference_model_1_layer_call_and_return_conditional_losses_39653?P-.4567HIOPQRcdjklm~??????????????????????????????B??
8?5
+?(
input_2???????????
p 

 
? "/?,
%?"
0???????????
? ?
B__inference_model_1_layer_call_and_return_conditional_losses_40572?P-.4567HIOPQRcdjklm~??????????????????????????????A?>
7?4
*?'
inputs???????????
p

 
? "/?,
%?"
0???????????
? ?
B__inference_model_1_layer_call_and_return_conditional_losses_40825?P-.4567HIOPQRcdjklm~??????????????????????????????A?>
7?4
*?'
inputs???????????
p 

 
? "/?,
%?"
0???????????
? ?
'__inference_model_1_layer_call_fn_39901?P-.4567HIOPQRcdjklm~??????????????????????????????B??
8?5
+?(
input_2???????????
p

 
? ""?????????????
'__inference_model_1_layer_call_fn_40148?P-.4567HIOPQRcdjklm~??????????????????????????????B??
8?5
+?(
input_2???????????
p 

 
? ""?????????????
'__inference_model_1_layer_call_fn_40930?P-.4567HIOPQRcdjklm~??????????????????????????????A?>
7?4
*?'
inputs???????????
p

 
? ""?????????????
'__inference_model_1_layer_call_fn_41035?P-.4567HIOPQRcdjklm~??????????????????????????????A?>
7?4
*?'
inputs???????????
p 

 
? ""?????????????
#__inference_signature_wrapper_40263?P-.4567HIOPQRcdjklm~??????????????????????????????E?B
? 
;?8
6
input_2+?(
input_2???????????"??<
:
	conv2d_29-?*
	conv2d_29???????????