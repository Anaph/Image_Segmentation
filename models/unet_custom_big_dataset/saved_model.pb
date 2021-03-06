??6
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
 ?"serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8??,
?
conv2d_46/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_46/kernel
}
$conv2d_46/kernel/Read/ReadVariableOpReadVariableOpconv2d_46/kernel*&
_output_shapes
: *
dtype0
t
conv2d_46/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_46/bias
m
"conv2d_46/bias/Read/ReadVariableOpReadVariableOpconv2d_46/bias*
_output_shapes
: *
dtype0
?
batch_normalization_43/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_43/gamma
?
0batch_normalization_43/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_43/gamma*
_output_shapes
: *
dtype0
?
batch_normalization_43/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_43/beta
?
/batch_normalization_43/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_43/beta*
_output_shapes
: *
dtype0
?
"batch_normalization_43/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_43/moving_mean
?
6batch_normalization_43/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_43/moving_mean*
_output_shapes
: *
dtype0
?
&batch_normalization_43/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_43/moving_variance
?
:batch_normalization_43/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_43/moving_variance*
_output_shapes
: *
dtype0
?
conv2d_48/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_48/kernel
}
$conv2d_48/kernel/Read/ReadVariableOpReadVariableOpconv2d_48/kernel*&
_output_shapes
: @*
dtype0
t
conv2d_48/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_48/bias
m
"conv2d_48/bias/Read/ReadVariableOpReadVariableOpconv2d_48/bias*
_output_shapes
:@*
dtype0
?
batch_normalization_45/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_45/gamma
?
0batch_normalization_45/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_45/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_45/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_45/beta
?
/batch_normalization_45/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_45/beta*
_output_shapes
:@*
dtype0
?
"batch_normalization_45/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_45/moving_mean
?
6batch_normalization_45/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_45/moving_mean*
_output_shapes
:@*
dtype0
?
&batch_normalization_45/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_45/moving_variance
?
:batch_normalization_45/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_45/moving_variance*
_output_shapes
:@*
dtype0
?
conv2d_50/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*!
shared_nameconv2d_50/kernel
~
$conv2d_50/kernel/Read/ReadVariableOpReadVariableOpconv2d_50/kernel*'
_output_shapes
:@?*
dtype0
u
conv2d_50/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_50/bias
n
"conv2d_50/bias/Read/ReadVariableOpReadVariableOpconv2d_50/bias*
_output_shapes	
:?*
dtype0
?
batch_normalization_47/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_47/gamma
?
0batch_normalization_47/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_47/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_47/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_47/beta
?
/batch_normalization_47/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_47/beta*
_output_shapes	
:?*
dtype0
?
"batch_normalization_47/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_47/moving_mean
?
6batch_normalization_47/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_47/moving_mean*
_output_shapes	
:?*
dtype0
?
&batch_normalization_47/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_47/moving_variance
?
:batch_normalization_47/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_47/moving_variance*
_output_shapes	
:?*
dtype0
?
conv2d_52/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*!
shared_nameconv2d_52/kernel

$conv2d_52/kernel/Read/ReadVariableOpReadVariableOpconv2d_52/kernel*(
_output_shapes
:??*
dtype0
u
conv2d_52/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_52/bias
n
"conv2d_52/bias/Read/ReadVariableOpReadVariableOpconv2d_52/bias*
_output_shapes	
:?*
dtype0
?
batch_normalization_49/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_49/gamma
?
0batch_normalization_49/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_49/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_49/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_49/beta
?
/batch_normalization_49/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_49/beta*
_output_shapes	
:?*
dtype0
?
"batch_normalization_49/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_49/moving_mean
?
6batch_normalization_49/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_49/moving_mean*
_output_shapes	
:?*
dtype0
?
&batch_normalization_49/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_49/moving_variance
?
:batch_normalization_49/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_49/moving_variance*
_output_shapes	
:?*
dtype0
?
conv2d_transpose_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??**
shared_nameconv2d_transpose_9/kernel
?
-conv2d_transpose_9/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_9/kernel*(
_output_shapes
:??*
dtype0
?
conv2d_transpose_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_nameconv2d_transpose_9/bias
?
+conv2d_transpose_9/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_9/bias*
_output_shapes	
:?*
dtype0
?
conv2d_54/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*!
shared_nameconv2d_54/kernel

$conv2d_54/kernel/Read/ReadVariableOpReadVariableOpconv2d_54/kernel*(
_output_shapes
:??*
dtype0
u
conv2d_54/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_54/bias
n
"conv2d_54/bias/Read/ReadVariableOpReadVariableOpconv2d_54/bias*
_output_shapes	
:?*
dtype0
?
batch_normalization_51/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_51/gamma
?
0batch_normalization_51/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_51/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_51/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_51/beta
?
/batch_normalization_51/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_51/beta*
_output_shapes	
:?*
dtype0
?
"batch_normalization_51/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_51/moving_mean
?
6batch_normalization_51/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_51/moving_mean*
_output_shapes	
:?*
dtype0
?
&batch_normalization_51/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_51/moving_variance
?
:batch_normalization_51/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_51/moving_variance*
_output_shapes	
:?*
dtype0
?
conv2d_transpose_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*+
shared_nameconv2d_transpose_10/kernel
?
.conv2d_transpose_10/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_10/kernel*'
_output_shapes
:@?*
dtype0
?
conv2d_transpose_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameconv2d_transpose_10/bias
?
,conv2d_transpose_10/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_10/bias*
_output_shapes
:@*
dtype0
?
conv2d_56/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?@*!
shared_nameconv2d_56/kernel
~
$conv2d_56/kernel/Read/ReadVariableOpReadVariableOpconv2d_56/kernel*'
_output_shapes
:?@*
dtype0
t
conv2d_56/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_56/bias
m
"conv2d_56/bias/Read/ReadVariableOpReadVariableOpconv2d_56/bias*
_output_shapes
:@*
dtype0
?
batch_normalization_53/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_53/gamma
?
0batch_normalization_53/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_53/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_53/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_53/beta
?
/batch_normalization_53/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_53/beta*
_output_shapes
:@*
dtype0
?
"batch_normalization_53/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_53/moving_mean
?
6batch_normalization_53/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_53/moving_mean*
_output_shapes
:@*
dtype0
?
&batch_normalization_53/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_53/moving_variance
?
:batch_normalization_53/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_53/moving_variance*
_output_shapes
:@*
dtype0
?
conv2d_transpose_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*+
shared_nameconv2d_transpose_11/kernel
?
.conv2d_transpose_11/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_11/kernel*&
_output_shapes
: @*
dtype0
?
conv2d_transpose_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameconv2d_transpose_11/bias
?
,conv2d_transpose_11/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_11/bias*
_output_shapes
: *
dtype0
?
conv2d_58/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *!
shared_nameconv2d_58/kernel
}
$conv2d_58/kernel/Read/ReadVariableOpReadVariableOpconv2d_58/kernel*&
_output_shapes
:@ *
dtype0
t
conv2d_58/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_58/bias
m
"conv2d_58/bias/Read/ReadVariableOpReadVariableOpconv2d_58/bias*
_output_shapes
: *
dtype0
?
batch_normalization_55/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_55/gamma
?
0batch_normalization_55/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_55/gamma*
_output_shapes
: *
dtype0
?
batch_normalization_55/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_55/beta
?
/batch_normalization_55/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_55/beta*
_output_shapes
: *
dtype0
?
"batch_normalization_55/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_55/moving_mean
?
6batch_normalization_55/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_55/moving_mean*
_output_shapes
: *
dtype0
?
&batch_normalization_55/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_55/moving_variance
?
:batch_normalization_55/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_55/moving_variance*
_output_shapes
: *
dtype0
?
conv2d_59/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_59/kernel
}
$conv2d_59/kernel/Read/ReadVariableOpReadVariableOpconv2d_59/kernel*&
_output_shapes
: *
dtype0
t
conv2d_59/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_59/bias
m
"conv2d_59/bias/Read/ReadVariableOpReadVariableOpconv2d_59/bias*
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
cond_1/Adam/conv2d_46/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name cond_1/Adam/conv2d_46/kernel/m
?
2cond_1/Adam/conv2d_46/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_46/kernel/m*&
_output_shapes
: *
dtype0
?
cond_1/Adam/conv2d_46/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namecond_1/Adam/conv2d_46/bias/m
?
0cond_1/Adam/conv2d_46/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_46/bias/m*
_output_shapes
: *
dtype0
?
*cond_1/Adam/batch_normalization_43/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*cond_1/Adam/batch_normalization_43/gamma/m
?
>cond_1/Adam/batch_normalization_43/gamma/m/Read/ReadVariableOpReadVariableOp*cond_1/Adam/batch_normalization_43/gamma/m*
_output_shapes
: *
dtype0
?
)cond_1/Adam/batch_normalization_43/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *:
shared_name+)cond_1/Adam/batch_normalization_43/beta/m
?
=cond_1/Adam/batch_normalization_43/beta/m/Read/ReadVariableOpReadVariableOp)cond_1/Adam/batch_normalization_43/beta/m*
_output_shapes
: *
dtype0
?
cond_1/Adam/conv2d_48/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*/
shared_name cond_1/Adam/conv2d_48/kernel/m
?
2cond_1/Adam/conv2d_48/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_48/kernel/m*&
_output_shapes
: @*
dtype0
?
cond_1/Adam/conv2d_48/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namecond_1/Adam/conv2d_48/bias/m
?
0cond_1/Adam/conv2d_48/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_48/bias/m*
_output_shapes
:@*
dtype0
?
*cond_1/Adam/batch_normalization_45/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*cond_1/Adam/batch_normalization_45/gamma/m
?
>cond_1/Adam/batch_normalization_45/gamma/m/Read/ReadVariableOpReadVariableOp*cond_1/Adam/batch_normalization_45/gamma/m*
_output_shapes
:@*
dtype0
?
)cond_1/Adam/batch_normalization_45/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)cond_1/Adam/batch_normalization_45/beta/m
?
=cond_1/Adam/batch_normalization_45/beta/m/Read/ReadVariableOpReadVariableOp)cond_1/Adam/batch_normalization_45/beta/m*
_output_shapes
:@*
dtype0
?
cond_1/Adam/conv2d_50/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*/
shared_name cond_1/Adam/conv2d_50/kernel/m
?
2cond_1/Adam/conv2d_50/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_50/kernel/m*'
_output_shapes
:@?*
dtype0
?
cond_1/Adam/conv2d_50/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namecond_1/Adam/conv2d_50/bias/m
?
0cond_1/Adam/conv2d_50/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_50/bias/m*
_output_shapes	
:?*
dtype0
?
*cond_1/Adam/batch_normalization_47/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*;
shared_name,*cond_1/Adam/batch_normalization_47/gamma/m
?
>cond_1/Adam/batch_normalization_47/gamma/m/Read/ReadVariableOpReadVariableOp*cond_1/Adam/batch_normalization_47/gamma/m*
_output_shapes	
:?*
dtype0
?
)cond_1/Adam/batch_normalization_47/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*:
shared_name+)cond_1/Adam/batch_normalization_47/beta/m
?
=cond_1/Adam/batch_normalization_47/beta/m/Read/ReadVariableOpReadVariableOp)cond_1/Adam/batch_normalization_47/beta/m*
_output_shapes	
:?*
dtype0
?
cond_1/Adam/conv2d_52/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*/
shared_name cond_1/Adam/conv2d_52/kernel/m
?
2cond_1/Adam/conv2d_52/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_52/kernel/m*(
_output_shapes
:??*
dtype0
?
cond_1/Adam/conv2d_52/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namecond_1/Adam/conv2d_52/bias/m
?
0cond_1/Adam/conv2d_52/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_52/bias/m*
_output_shapes	
:?*
dtype0
?
*cond_1/Adam/batch_normalization_49/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*;
shared_name,*cond_1/Adam/batch_normalization_49/gamma/m
?
>cond_1/Adam/batch_normalization_49/gamma/m/Read/ReadVariableOpReadVariableOp*cond_1/Adam/batch_normalization_49/gamma/m*
_output_shapes	
:?*
dtype0
?
)cond_1/Adam/batch_normalization_49/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*:
shared_name+)cond_1/Adam/batch_normalization_49/beta/m
?
=cond_1/Adam/batch_normalization_49/beta/m/Read/ReadVariableOpReadVariableOp)cond_1/Adam/batch_normalization_49/beta/m*
_output_shapes	
:?*
dtype0
?
'cond_1/Adam/conv2d_transpose_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*8
shared_name)'cond_1/Adam/conv2d_transpose_9/kernel/m
?
;cond_1/Adam/conv2d_transpose_9/kernel/m/Read/ReadVariableOpReadVariableOp'cond_1/Adam/conv2d_transpose_9/kernel/m*(
_output_shapes
:??*
dtype0
?
%cond_1/Adam/conv2d_transpose_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*6
shared_name'%cond_1/Adam/conv2d_transpose_9/bias/m
?
9cond_1/Adam/conv2d_transpose_9/bias/m/Read/ReadVariableOpReadVariableOp%cond_1/Adam/conv2d_transpose_9/bias/m*
_output_shapes	
:?*
dtype0
?
cond_1/Adam/conv2d_54/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*/
shared_name cond_1/Adam/conv2d_54/kernel/m
?
2cond_1/Adam/conv2d_54/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_54/kernel/m*(
_output_shapes
:??*
dtype0
?
cond_1/Adam/conv2d_54/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namecond_1/Adam/conv2d_54/bias/m
?
0cond_1/Adam/conv2d_54/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_54/bias/m*
_output_shapes	
:?*
dtype0
?
*cond_1/Adam/batch_normalization_51/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*;
shared_name,*cond_1/Adam/batch_normalization_51/gamma/m
?
>cond_1/Adam/batch_normalization_51/gamma/m/Read/ReadVariableOpReadVariableOp*cond_1/Adam/batch_normalization_51/gamma/m*
_output_shapes	
:?*
dtype0
?
)cond_1/Adam/batch_normalization_51/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*:
shared_name+)cond_1/Adam/batch_normalization_51/beta/m
?
=cond_1/Adam/batch_normalization_51/beta/m/Read/ReadVariableOpReadVariableOp)cond_1/Adam/batch_normalization_51/beta/m*
_output_shapes	
:?*
dtype0
?
(cond_1/Adam/conv2d_transpose_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*9
shared_name*(cond_1/Adam/conv2d_transpose_10/kernel/m
?
<cond_1/Adam/conv2d_transpose_10/kernel/m/Read/ReadVariableOpReadVariableOp(cond_1/Adam/conv2d_transpose_10/kernel/m*'
_output_shapes
:@?*
dtype0
?
&cond_1/Adam/conv2d_transpose_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&cond_1/Adam/conv2d_transpose_10/bias/m
?
:cond_1/Adam/conv2d_transpose_10/bias/m/Read/ReadVariableOpReadVariableOp&cond_1/Adam/conv2d_transpose_10/bias/m*
_output_shapes
:@*
dtype0
?
cond_1/Adam/conv2d_56/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?@*/
shared_name cond_1/Adam/conv2d_56/kernel/m
?
2cond_1/Adam/conv2d_56/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_56/kernel/m*'
_output_shapes
:?@*
dtype0
?
cond_1/Adam/conv2d_56/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namecond_1/Adam/conv2d_56/bias/m
?
0cond_1/Adam/conv2d_56/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_56/bias/m*
_output_shapes
:@*
dtype0
?
*cond_1/Adam/batch_normalization_53/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*cond_1/Adam/batch_normalization_53/gamma/m
?
>cond_1/Adam/batch_normalization_53/gamma/m/Read/ReadVariableOpReadVariableOp*cond_1/Adam/batch_normalization_53/gamma/m*
_output_shapes
:@*
dtype0
?
)cond_1/Adam/batch_normalization_53/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)cond_1/Adam/batch_normalization_53/beta/m
?
=cond_1/Adam/batch_normalization_53/beta/m/Read/ReadVariableOpReadVariableOp)cond_1/Adam/batch_normalization_53/beta/m*
_output_shapes
:@*
dtype0
?
(cond_1/Adam/conv2d_transpose_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*9
shared_name*(cond_1/Adam/conv2d_transpose_11/kernel/m
?
<cond_1/Adam/conv2d_transpose_11/kernel/m/Read/ReadVariableOpReadVariableOp(cond_1/Adam/conv2d_transpose_11/kernel/m*&
_output_shapes
: @*
dtype0
?
&cond_1/Adam/conv2d_transpose_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&cond_1/Adam/conv2d_transpose_11/bias/m
?
:cond_1/Adam/conv2d_transpose_11/bias/m/Read/ReadVariableOpReadVariableOp&cond_1/Adam/conv2d_transpose_11/bias/m*
_output_shapes
: *
dtype0
?
cond_1/Adam/conv2d_58/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ */
shared_name cond_1/Adam/conv2d_58/kernel/m
?
2cond_1/Adam/conv2d_58/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_58/kernel/m*&
_output_shapes
:@ *
dtype0
?
cond_1/Adam/conv2d_58/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namecond_1/Adam/conv2d_58/bias/m
?
0cond_1/Adam/conv2d_58/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_58/bias/m*
_output_shapes
: *
dtype0
?
*cond_1/Adam/batch_normalization_55/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*cond_1/Adam/batch_normalization_55/gamma/m
?
>cond_1/Adam/batch_normalization_55/gamma/m/Read/ReadVariableOpReadVariableOp*cond_1/Adam/batch_normalization_55/gamma/m*
_output_shapes
: *
dtype0
?
)cond_1/Adam/batch_normalization_55/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *:
shared_name+)cond_1/Adam/batch_normalization_55/beta/m
?
=cond_1/Adam/batch_normalization_55/beta/m/Read/ReadVariableOpReadVariableOp)cond_1/Adam/batch_normalization_55/beta/m*
_output_shapes
: *
dtype0
?
cond_1/Adam/conv2d_59/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name cond_1/Adam/conv2d_59/kernel/m
?
2cond_1/Adam/conv2d_59/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_59/kernel/m*&
_output_shapes
: *
dtype0
?
cond_1/Adam/conv2d_59/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namecond_1/Adam/conv2d_59/bias/m
?
0cond_1/Adam/conv2d_59/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_59/bias/m*
_output_shapes
:*
dtype0
?
cond_1/Adam/conv2d_46/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name cond_1/Adam/conv2d_46/kernel/v
?
2cond_1/Adam/conv2d_46/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_46/kernel/v*&
_output_shapes
: *
dtype0
?
cond_1/Adam/conv2d_46/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namecond_1/Adam/conv2d_46/bias/v
?
0cond_1/Adam/conv2d_46/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_46/bias/v*
_output_shapes
: *
dtype0
?
*cond_1/Adam/batch_normalization_43/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*cond_1/Adam/batch_normalization_43/gamma/v
?
>cond_1/Adam/batch_normalization_43/gamma/v/Read/ReadVariableOpReadVariableOp*cond_1/Adam/batch_normalization_43/gamma/v*
_output_shapes
: *
dtype0
?
)cond_1/Adam/batch_normalization_43/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *:
shared_name+)cond_1/Adam/batch_normalization_43/beta/v
?
=cond_1/Adam/batch_normalization_43/beta/v/Read/ReadVariableOpReadVariableOp)cond_1/Adam/batch_normalization_43/beta/v*
_output_shapes
: *
dtype0
?
cond_1/Adam/conv2d_48/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*/
shared_name cond_1/Adam/conv2d_48/kernel/v
?
2cond_1/Adam/conv2d_48/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_48/kernel/v*&
_output_shapes
: @*
dtype0
?
cond_1/Adam/conv2d_48/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namecond_1/Adam/conv2d_48/bias/v
?
0cond_1/Adam/conv2d_48/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_48/bias/v*
_output_shapes
:@*
dtype0
?
*cond_1/Adam/batch_normalization_45/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*cond_1/Adam/batch_normalization_45/gamma/v
?
>cond_1/Adam/batch_normalization_45/gamma/v/Read/ReadVariableOpReadVariableOp*cond_1/Adam/batch_normalization_45/gamma/v*
_output_shapes
:@*
dtype0
?
)cond_1/Adam/batch_normalization_45/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)cond_1/Adam/batch_normalization_45/beta/v
?
=cond_1/Adam/batch_normalization_45/beta/v/Read/ReadVariableOpReadVariableOp)cond_1/Adam/batch_normalization_45/beta/v*
_output_shapes
:@*
dtype0
?
cond_1/Adam/conv2d_50/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*/
shared_name cond_1/Adam/conv2d_50/kernel/v
?
2cond_1/Adam/conv2d_50/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_50/kernel/v*'
_output_shapes
:@?*
dtype0
?
cond_1/Adam/conv2d_50/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namecond_1/Adam/conv2d_50/bias/v
?
0cond_1/Adam/conv2d_50/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_50/bias/v*
_output_shapes	
:?*
dtype0
?
*cond_1/Adam/batch_normalization_47/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*;
shared_name,*cond_1/Adam/batch_normalization_47/gamma/v
?
>cond_1/Adam/batch_normalization_47/gamma/v/Read/ReadVariableOpReadVariableOp*cond_1/Adam/batch_normalization_47/gamma/v*
_output_shapes	
:?*
dtype0
?
)cond_1/Adam/batch_normalization_47/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*:
shared_name+)cond_1/Adam/batch_normalization_47/beta/v
?
=cond_1/Adam/batch_normalization_47/beta/v/Read/ReadVariableOpReadVariableOp)cond_1/Adam/batch_normalization_47/beta/v*
_output_shapes	
:?*
dtype0
?
cond_1/Adam/conv2d_52/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*/
shared_name cond_1/Adam/conv2d_52/kernel/v
?
2cond_1/Adam/conv2d_52/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_52/kernel/v*(
_output_shapes
:??*
dtype0
?
cond_1/Adam/conv2d_52/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namecond_1/Adam/conv2d_52/bias/v
?
0cond_1/Adam/conv2d_52/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_52/bias/v*
_output_shapes	
:?*
dtype0
?
*cond_1/Adam/batch_normalization_49/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*;
shared_name,*cond_1/Adam/batch_normalization_49/gamma/v
?
>cond_1/Adam/batch_normalization_49/gamma/v/Read/ReadVariableOpReadVariableOp*cond_1/Adam/batch_normalization_49/gamma/v*
_output_shapes	
:?*
dtype0
?
)cond_1/Adam/batch_normalization_49/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*:
shared_name+)cond_1/Adam/batch_normalization_49/beta/v
?
=cond_1/Adam/batch_normalization_49/beta/v/Read/ReadVariableOpReadVariableOp)cond_1/Adam/batch_normalization_49/beta/v*
_output_shapes	
:?*
dtype0
?
'cond_1/Adam/conv2d_transpose_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*8
shared_name)'cond_1/Adam/conv2d_transpose_9/kernel/v
?
;cond_1/Adam/conv2d_transpose_9/kernel/v/Read/ReadVariableOpReadVariableOp'cond_1/Adam/conv2d_transpose_9/kernel/v*(
_output_shapes
:??*
dtype0
?
%cond_1/Adam/conv2d_transpose_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*6
shared_name'%cond_1/Adam/conv2d_transpose_9/bias/v
?
9cond_1/Adam/conv2d_transpose_9/bias/v/Read/ReadVariableOpReadVariableOp%cond_1/Adam/conv2d_transpose_9/bias/v*
_output_shapes	
:?*
dtype0
?
cond_1/Adam/conv2d_54/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*/
shared_name cond_1/Adam/conv2d_54/kernel/v
?
2cond_1/Adam/conv2d_54/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_54/kernel/v*(
_output_shapes
:??*
dtype0
?
cond_1/Adam/conv2d_54/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namecond_1/Adam/conv2d_54/bias/v
?
0cond_1/Adam/conv2d_54/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_54/bias/v*
_output_shapes	
:?*
dtype0
?
*cond_1/Adam/batch_normalization_51/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*;
shared_name,*cond_1/Adam/batch_normalization_51/gamma/v
?
>cond_1/Adam/batch_normalization_51/gamma/v/Read/ReadVariableOpReadVariableOp*cond_1/Adam/batch_normalization_51/gamma/v*
_output_shapes	
:?*
dtype0
?
)cond_1/Adam/batch_normalization_51/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*:
shared_name+)cond_1/Adam/batch_normalization_51/beta/v
?
=cond_1/Adam/batch_normalization_51/beta/v/Read/ReadVariableOpReadVariableOp)cond_1/Adam/batch_normalization_51/beta/v*
_output_shapes	
:?*
dtype0
?
(cond_1/Adam/conv2d_transpose_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*9
shared_name*(cond_1/Adam/conv2d_transpose_10/kernel/v
?
<cond_1/Adam/conv2d_transpose_10/kernel/v/Read/ReadVariableOpReadVariableOp(cond_1/Adam/conv2d_transpose_10/kernel/v*'
_output_shapes
:@?*
dtype0
?
&cond_1/Adam/conv2d_transpose_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&cond_1/Adam/conv2d_transpose_10/bias/v
?
:cond_1/Adam/conv2d_transpose_10/bias/v/Read/ReadVariableOpReadVariableOp&cond_1/Adam/conv2d_transpose_10/bias/v*
_output_shapes
:@*
dtype0
?
cond_1/Adam/conv2d_56/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?@*/
shared_name cond_1/Adam/conv2d_56/kernel/v
?
2cond_1/Adam/conv2d_56/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_56/kernel/v*'
_output_shapes
:?@*
dtype0
?
cond_1/Adam/conv2d_56/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namecond_1/Adam/conv2d_56/bias/v
?
0cond_1/Adam/conv2d_56/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_56/bias/v*
_output_shapes
:@*
dtype0
?
*cond_1/Adam/batch_normalization_53/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*cond_1/Adam/batch_normalization_53/gamma/v
?
>cond_1/Adam/batch_normalization_53/gamma/v/Read/ReadVariableOpReadVariableOp*cond_1/Adam/batch_normalization_53/gamma/v*
_output_shapes
:@*
dtype0
?
)cond_1/Adam/batch_normalization_53/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)cond_1/Adam/batch_normalization_53/beta/v
?
=cond_1/Adam/batch_normalization_53/beta/v/Read/ReadVariableOpReadVariableOp)cond_1/Adam/batch_normalization_53/beta/v*
_output_shapes
:@*
dtype0
?
(cond_1/Adam/conv2d_transpose_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*9
shared_name*(cond_1/Adam/conv2d_transpose_11/kernel/v
?
<cond_1/Adam/conv2d_transpose_11/kernel/v/Read/ReadVariableOpReadVariableOp(cond_1/Adam/conv2d_transpose_11/kernel/v*&
_output_shapes
: @*
dtype0
?
&cond_1/Adam/conv2d_transpose_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&cond_1/Adam/conv2d_transpose_11/bias/v
?
:cond_1/Adam/conv2d_transpose_11/bias/v/Read/ReadVariableOpReadVariableOp&cond_1/Adam/conv2d_transpose_11/bias/v*
_output_shapes
: *
dtype0
?
cond_1/Adam/conv2d_58/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ */
shared_name cond_1/Adam/conv2d_58/kernel/v
?
2cond_1/Adam/conv2d_58/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_58/kernel/v*&
_output_shapes
:@ *
dtype0
?
cond_1/Adam/conv2d_58/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namecond_1/Adam/conv2d_58/bias/v
?
0cond_1/Adam/conv2d_58/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_58/bias/v*
_output_shapes
: *
dtype0
?
*cond_1/Adam/batch_normalization_55/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*cond_1/Adam/batch_normalization_55/gamma/v
?
>cond_1/Adam/batch_normalization_55/gamma/v/Read/ReadVariableOpReadVariableOp*cond_1/Adam/batch_normalization_55/gamma/v*
_output_shapes
: *
dtype0
?
)cond_1/Adam/batch_normalization_55/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *:
shared_name+)cond_1/Adam/batch_normalization_55/beta/v
?
=cond_1/Adam/batch_normalization_55/beta/v/Read/ReadVariableOpReadVariableOp)cond_1/Adam/batch_normalization_55/beta/v*
_output_shapes
: *
dtype0
?
cond_1/Adam/conv2d_59/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name cond_1/Adam/conv2d_59/kernel/v
?
2cond_1/Adam/conv2d_59/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_59/kernel/v*&
_output_shapes
: *
dtype0
?
cond_1/Adam/conv2d_59/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namecond_1/Adam/conv2d_59/bias/v
?
0cond_1/Adam/conv2d_59/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2d_59/bias/v*
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
/	variables
0trainable_variables
1regularization_losses
2	keras_api
?
3axis
	4gamma
5beta
6moving_mean
7moving_variance
8	variables
9trainable_variables
:regularization_losses
;	keras_api
R
<	variables
=trainable_variables
>regularization_losses
?	keras_api
R
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
R
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
h

Hkernel
Ibias
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
?
Naxis
	Ogamma
Pbeta
Qmoving_mean
Rmoving_variance
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
R
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
R
[	variables
\trainable_variables
]regularization_losses
^	keras_api
R
_	variables
`trainable_variables
aregularization_losses
b	keras_api
h

ckernel
dbias
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
?
iaxis
	jgamma
kbeta
lmoving_mean
mmoving_variance
n	variables
otrainable_variables
pregularization_losses
q	keras_api
R
r	variables
strainable_variables
tregularization_losses
u	keras_api
R
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
R
z	variables
{trainable_variables
|regularization_losses
}	keras_api
l

~kernel
bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
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
?layer_metrics
?non_trainable_variables
?metrics
(	variables
 ?layer_regularization_losses
?layers
)trainable_variables
*regularization_losses
 
\Z
VARIABLE_VALUEconv2d_46/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_46/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

-0
.1

-0
.1
 
?
?layer_metrics
?non_trainable_variables
?metrics
/	variables
 ?layer_regularization_losses
?layers
0trainable_variables
1regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_43/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_43/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_43/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_43/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

40
51
62
73

40
51
 
?
?layer_metrics
?non_trainable_variables
?metrics
8	variables
 ?layer_regularization_losses
?layers
9trainable_variables
:regularization_losses
 
 
 
?
?layer_metrics
?non_trainable_variables
?metrics
<	variables
 ?layer_regularization_losses
?layers
=trainable_variables
>regularization_losses
 
 
 
?
?layer_metrics
?non_trainable_variables
?metrics
@	variables
 ?layer_regularization_losses
?layers
Atrainable_variables
Bregularization_losses
 
 
 
?
?layer_metrics
?non_trainable_variables
?metrics
D	variables
 ?layer_regularization_losses
?layers
Etrainable_variables
Fregularization_losses
\Z
VARIABLE_VALUEconv2d_48/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_48/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

H0
I1

H0
I1
 
?
?layer_metrics
?non_trainable_variables
?metrics
J	variables
 ?layer_regularization_losses
?layers
Ktrainable_variables
Lregularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_45/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_45/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_45/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_45/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

O0
P1
Q2
R3

O0
P1
 
?
?layer_metrics
?non_trainable_variables
?metrics
S	variables
 ?layer_regularization_losses
?layers
Ttrainable_variables
Uregularization_losses
 
 
 
?
?layer_metrics
?non_trainable_variables
?metrics
W	variables
 ?layer_regularization_losses
?layers
Xtrainable_variables
Yregularization_losses
 
 
 
?
?layer_metrics
?non_trainable_variables
?metrics
[	variables
 ?layer_regularization_losses
?layers
\trainable_variables
]regularization_losses
 
 
 
?
?layer_metrics
?non_trainable_variables
?metrics
_	variables
 ?layer_regularization_losses
?layers
`trainable_variables
aregularization_losses
\Z
VARIABLE_VALUEconv2d_50/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_50/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

c0
d1

c0
d1
 
?
?layer_metrics
?non_trainable_variables
?metrics
e	variables
 ?layer_regularization_losses
?layers
ftrainable_variables
gregularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_47/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_47/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_47/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_47/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

j0
k1
l2
m3

j0
k1
 
?
?layer_metrics
?non_trainable_variables
?metrics
n	variables
 ?layer_regularization_losses
?layers
otrainable_variables
pregularization_losses
 
 
 
?
?layer_metrics
?non_trainable_variables
?metrics
r	variables
 ?layer_regularization_losses
?layers
strainable_variables
tregularization_losses
 
 
 
?
?layer_metrics
?non_trainable_variables
?metrics
v	variables
 ?layer_regularization_losses
?layers
wtrainable_variables
xregularization_losses
 
 
 
?
?layer_metrics
?non_trainable_variables
?metrics
z	variables
 ?layer_regularization_losses
?layers
{trainable_variables
|regularization_losses
\Z
VARIABLE_VALUEconv2d_52/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_52/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

~0
1

~0
1
 
?
?layer_metrics
?non_trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_49/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_49/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_49/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_49/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
?0
?1
?2
?3

?0
?1
 
?
?layer_metrics
?non_trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
 
 
 
?
?layer_metrics
?non_trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
ec
VARIABLE_VALUEconv2d_transpose_9/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEconv2d_transpose_9/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?layer_metrics
?non_trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
 
 
 
?
?layer_metrics
?non_trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
 
 
 
?
?layer_metrics
?non_trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
\Z
VARIABLE_VALUEconv2d_54/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_54/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?layer_metrics
?non_trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
 
hf
VARIABLE_VALUEbatch_normalization_51/gamma6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_51/beta5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_51/moving_mean<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_51/moving_variance@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
?0
?1
?2
?3

?0
?1
 
?
?layer_metrics
?non_trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
 
 
 
?
?layer_metrics
?non_trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
ge
VARIABLE_VALUEconv2d_transpose_10/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEconv2d_transpose_10/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?layer_metrics
?non_trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
 
 
 
?
?layer_metrics
?non_trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
 
 
 
?
?layer_metrics
?non_trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
][
VARIABLE_VALUEconv2d_56/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_56/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?layer_metrics
?non_trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
 
hf
VARIABLE_VALUEbatch_normalization_53/gamma6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_53/beta5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_53/moving_mean<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_53/moving_variance@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
?0
?1
?2
?3

?0
?1
 
?
?layer_metrics
?non_trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
 
 
 
?
?layer_metrics
?non_trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
ge
VARIABLE_VALUEconv2d_transpose_11/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEconv2d_transpose_11/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?layer_metrics
?non_trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
 
 
 
?
?layer_metrics
?non_trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
 
 
 
?
?layer_metrics
?non_trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
][
VARIABLE_VALUEconv2d_58/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_58/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?layer_metrics
?non_trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
 
hf
VARIABLE_VALUEbatch_normalization_55/gamma6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_55/beta5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_55/moving_mean<layer_with_weights-16/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_55/moving_variance@layer_with_weights-16/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
?0
?1
?2
?3

?0
?1
 
?
?layer_metrics
?non_trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
 
 
 
?
?layer_metrics
?non_trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
][
VARIABLE_VALUEconv2d_59/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_59/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?layer_metrics
?non_trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
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
VARIABLE_VALUEcond_1/Adam/conv2d_46/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_46/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*cond_1/Adam/batch_normalization_43/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)cond_1/Adam/batch_normalization_43/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_48/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_48/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*cond_1/Adam/batch_normalization_45/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)cond_1/Adam/batch_normalization_45/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_50/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_50/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*cond_1/Adam/batch_normalization_47/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)cond_1/Adam/batch_normalization_47/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_52/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_52/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*cond_1/Adam/batch_normalization_49/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)cond_1/Adam/batch_normalization_49/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE'cond_1/Adam/conv2d_transpose_9/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE%cond_1/Adam/conv2d_transpose_9/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_54/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_54/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*cond_1/Adam/batch_normalization_51/gamma/mRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)cond_1/Adam/batch_normalization_51/beta/mQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE(cond_1/Adam/conv2d_transpose_10/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE&cond_1/Adam/conv2d_transpose_10/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_56/kernel/mSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_56/bias/mQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*cond_1/Adam/batch_normalization_53/gamma/mRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)cond_1/Adam/batch_normalization_53/beta/mQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE(cond_1/Adam/conv2d_transpose_11/kernel/mSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE&cond_1/Adam/conv2d_transpose_11/bias/mQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_58/kernel/mSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_58/bias/mQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*cond_1/Adam/batch_normalization_55/gamma/mRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)cond_1/Adam/batch_normalization_55/beta/mQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_59/kernel/mSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_59/bias/mQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_46/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_46/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*cond_1/Adam/batch_normalization_43/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)cond_1/Adam/batch_normalization_43/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_48/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_48/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*cond_1/Adam/batch_normalization_45/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)cond_1/Adam/batch_normalization_45/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_50/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_50/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*cond_1/Adam/batch_normalization_47/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)cond_1/Adam/batch_normalization_47/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_52/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_52/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*cond_1/Adam/batch_normalization_49/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)cond_1/Adam/batch_normalization_49/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE'cond_1/Adam/conv2d_transpose_9/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE%cond_1/Adam/conv2d_transpose_9/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_54/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_54/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*cond_1/Adam/batch_normalization_51/gamma/vRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)cond_1/Adam/batch_normalization_51/beta/vQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE(cond_1/Adam/conv2d_transpose_10/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE&cond_1/Adam/conv2d_transpose_10/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_56/kernel/vSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_56/bias/vQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*cond_1/Adam/batch_normalization_53/gamma/vRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)cond_1/Adam/batch_normalization_53/beta/vQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE(cond_1/Adam/conv2d_transpose_11/kernel/vSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE&cond_1/Adam/conv2d_transpose_11/bias/vQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_58/kernel/vSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_58/bias/vQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*cond_1/Adam/batch_normalization_55/gamma/vRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)cond_1/Adam/batch_normalization_55/beta/vQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_59/kernel/vSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEcond_1/Adam/conv2d_59/bias/vQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_4Placeholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_4conv2d_46/kernelconv2d_46/biasbatch_normalization_43/gammabatch_normalization_43/beta"batch_normalization_43/moving_mean&batch_normalization_43/moving_varianceconv2d_48/kernelconv2d_48/biasbatch_normalization_45/gammabatch_normalization_45/beta"batch_normalization_45/moving_mean&batch_normalization_45/moving_varianceconv2d_50/kernelconv2d_50/biasbatch_normalization_47/gammabatch_normalization_47/beta"batch_normalization_47/moving_mean&batch_normalization_47/moving_varianceconv2d_52/kernelconv2d_52/biasbatch_normalization_49/gammabatch_normalization_49/beta"batch_normalization_49/moving_mean&batch_normalization_49/moving_varianceconv2d_transpose_9/kernelconv2d_transpose_9/biasconv2d_54/kernelconv2d_54/biasbatch_normalization_51/gammabatch_normalization_51/beta"batch_normalization_51/moving_mean&batch_normalization_51/moving_varianceconv2d_transpose_10/kernelconv2d_transpose_10/biasconv2d_56/kernelconv2d_56/biasbatch_normalization_53/gammabatch_normalization_53/beta"batch_normalization_53/moving_mean&batch_normalization_53/moving_varianceconv2d_transpose_11/kernelconv2d_transpose_11/biasconv2d_58/kernelconv2d_58/biasbatch_normalization_55/gammabatch_normalization_55/beta"batch_normalization_55/moving_mean&batch_normalization_55/moving_varianceconv2d_59/kernelconv2d_59/bias*>
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
#__inference_signature_wrapper_79058
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?9
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_46/kernel/Read/ReadVariableOp"conv2d_46/bias/Read/ReadVariableOp0batch_normalization_43/gamma/Read/ReadVariableOp/batch_normalization_43/beta/Read/ReadVariableOp6batch_normalization_43/moving_mean/Read/ReadVariableOp:batch_normalization_43/moving_variance/Read/ReadVariableOp$conv2d_48/kernel/Read/ReadVariableOp"conv2d_48/bias/Read/ReadVariableOp0batch_normalization_45/gamma/Read/ReadVariableOp/batch_normalization_45/beta/Read/ReadVariableOp6batch_normalization_45/moving_mean/Read/ReadVariableOp:batch_normalization_45/moving_variance/Read/ReadVariableOp$conv2d_50/kernel/Read/ReadVariableOp"conv2d_50/bias/Read/ReadVariableOp0batch_normalization_47/gamma/Read/ReadVariableOp/batch_normalization_47/beta/Read/ReadVariableOp6batch_normalization_47/moving_mean/Read/ReadVariableOp:batch_normalization_47/moving_variance/Read/ReadVariableOp$conv2d_52/kernel/Read/ReadVariableOp"conv2d_52/bias/Read/ReadVariableOp0batch_normalization_49/gamma/Read/ReadVariableOp/batch_normalization_49/beta/Read/ReadVariableOp6batch_normalization_49/moving_mean/Read/ReadVariableOp:batch_normalization_49/moving_variance/Read/ReadVariableOp-conv2d_transpose_9/kernel/Read/ReadVariableOp+conv2d_transpose_9/bias/Read/ReadVariableOp$conv2d_54/kernel/Read/ReadVariableOp"conv2d_54/bias/Read/ReadVariableOp0batch_normalization_51/gamma/Read/ReadVariableOp/batch_normalization_51/beta/Read/ReadVariableOp6batch_normalization_51/moving_mean/Read/ReadVariableOp:batch_normalization_51/moving_variance/Read/ReadVariableOp.conv2d_transpose_10/kernel/Read/ReadVariableOp,conv2d_transpose_10/bias/Read/ReadVariableOp$conv2d_56/kernel/Read/ReadVariableOp"conv2d_56/bias/Read/ReadVariableOp0batch_normalization_53/gamma/Read/ReadVariableOp/batch_normalization_53/beta/Read/ReadVariableOp6batch_normalization_53/moving_mean/Read/ReadVariableOp:batch_normalization_53/moving_variance/Read/ReadVariableOp.conv2d_transpose_11/kernel/Read/ReadVariableOp,conv2d_transpose_11/bias/Read/ReadVariableOp$conv2d_58/kernel/Read/ReadVariableOp"conv2d_58/bias/Read/ReadVariableOp0batch_normalization_55/gamma/Read/ReadVariableOp/batch_normalization_55/beta/Read/ReadVariableOp6batch_normalization_55/moving_mean/Read/ReadVariableOp:batch_normalization_55/moving_variance/Read/ReadVariableOp$conv2d_59/kernel/Read/ReadVariableOp"conv2d_59/bias/Read/ReadVariableOp$cond_1/Adam/iter/Read/ReadVariableOp&cond_1/Adam/beta_1/Read/ReadVariableOp&cond_1/Adam/beta_2/Read/ReadVariableOp%cond_1/Adam/decay/Read/ReadVariableOp-cond_1/Adam/learning_rate/Read/ReadVariableOp&current_loss_scale/Read/ReadVariableOpgood_steps/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp2cond_1/Adam/conv2d_46/kernel/m/Read/ReadVariableOp0cond_1/Adam/conv2d_46/bias/m/Read/ReadVariableOp>cond_1/Adam/batch_normalization_43/gamma/m/Read/ReadVariableOp=cond_1/Adam/batch_normalization_43/beta/m/Read/ReadVariableOp2cond_1/Adam/conv2d_48/kernel/m/Read/ReadVariableOp0cond_1/Adam/conv2d_48/bias/m/Read/ReadVariableOp>cond_1/Adam/batch_normalization_45/gamma/m/Read/ReadVariableOp=cond_1/Adam/batch_normalization_45/beta/m/Read/ReadVariableOp2cond_1/Adam/conv2d_50/kernel/m/Read/ReadVariableOp0cond_1/Adam/conv2d_50/bias/m/Read/ReadVariableOp>cond_1/Adam/batch_normalization_47/gamma/m/Read/ReadVariableOp=cond_1/Adam/batch_normalization_47/beta/m/Read/ReadVariableOp2cond_1/Adam/conv2d_52/kernel/m/Read/ReadVariableOp0cond_1/Adam/conv2d_52/bias/m/Read/ReadVariableOp>cond_1/Adam/batch_normalization_49/gamma/m/Read/ReadVariableOp=cond_1/Adam/batch_normalization_49/beta/m/Read/ReadVariableOp;cond_1/Adam/conv2d_transpose_9/kernel/m/Read/ReadVariableOp9cond_1/Adam/conv2d_transpose_9/bias/m/Read/ReadVariableOp2cond_1/Adam/conv2d_54/kernel/m/Read/ReadVariableOp0cond_1/Adam/conv2d_54/bias/m/Read/ReadVariableOp>cond_1/Adam/batch_normalization_51/gamma/m/Read/ReadVariableOp=cond_1/Adam/batch_normalization_51/beta/m/Read/ReadVariableOp<cond_1/Adam/conv2d_transpose_10/kernel/m/Read/ReadVariableOp:cond_1/Adam/conv2d_transpose_10/bias/m/Read/ReadVariableOp2cond_1/Adam/conv2d_56/kernel/m/Read/ReadVariableOp0cond_1/Adam/conv2d_56/bias/m/Read/ReadVariableOp>cond_1/Adam/batch_normalization_53/gamma/m/Read/ReadVariableOp=cond_1/Adam/batch_normalization_53/beta/m/Read/ReadVariableOp<cond_1/Adam/conv2d_transpose_11/kernel/m/Read/ReadVariableOp:cond_1/Adam/conv2d_transpose_11/bias/m/Read/ReadVariableOp2cond_1/Adam/conv2d_58/kernel/m/Read/ReadVariableOp0cond_1/Adam/conv2d_58/bias/m/Read/ReadVariableOp>cond_1/Adam/batch_normalization_55/gamma/m/Read/ReadVariableOp=cond_1/Adam/batch_normalization_55/beta/m/Read/ReadVariableOp2cond_1/Adam/conv2d_59/kernel/m/Read/ReadVariableOp0cond_1/Adam/conv2d_59/bias/m/Read/ReadVariableOp2cond_1/Adam/conv2d_46/kernel/v/Read/ReadVariableOp0cond_1/Adam/conv2d_46/bias/v/Read/ReadVariableOp>cond_1/Adam/batch_normalization_43/gamma/v/Read/ReadVariableOp=cond_1/Adam/batch_normalization_43/beta/v/Read/ReadVariableOp2cond_1/Adam/conv2d_48/kernel/v/Read/ReadVariableOp0cond_1/Adam/conv2d_48/bias/v/Read/ReadVariableOp>cond_1/Adam/batch_normalization_45/gamma/v/Read/ReadVariableOp=cond_1/Adam/batch_normalization_45/beta/v/Read/ReadVariableOp2cond_1/Adam/conv2d_50/kernel/v/Read/ReadVariableOp0cond_1/Adam/conv2d_50/bias/v/Read/ReadVariableOp>cond_1/Adam/batch_normalization_47/gamma/v/Read/ReadVariableOp=cond_1/Adam/batch_normalization_47/beta/v/Read/ReadVariableOp2cond_1/Adam/conv2d_52/kernel/v/Read/ReadVariableOp0cond_1/Adam/conv2d_52/bias/v/Read/ReadVariableOp>cond_1/Adam/batch_normalization_49/gamma/v/Read/ReadVariableOp=cond_1/Adam/batch_normalization_49/beta/v/Read/ReadVariableOp;cond_1/Adam/conv2d_transpose_9/kernel/v/Read/ReadVariableOp9cond_1/Adam/conv2d_transpose_9/bias/v/Read/ReadVariableOp2cond_1/Adam/conv2d_54/kernel/v/Read/ReadVariableOp0cond_1/Adam/conv2d_54/bias/v/Read/ReadVariableOp>cond_1/Adam/batch_normalization_51/gamma/v/Read/ReadVariableOp=cond_1/Adam/batch_normalization_51/beta/v/Read/ReadVariableOp<cond_1/Adam/conv2d_transpose_10/kernel/v/Read/ReadVariableOp:cond_1/Adam/conv2d_transpose_10/bias/v/Read/ReadVariableOp2cond_1/Adam/conv2d_56/kernel/v/Read/ReadVariableOp0cond_1/Adam/conv2d_56/bias/v/Read/ReadVariableOp>cond_1/Adam/batch_normalization_53/gamma/v/Read/ReadVariableOp=cond_1/Adam/batch_normalization_53/beta/v/Read/ReadVariableOp<cond_1/Adam/conv2d_transpose_11/kernel/v/Read/ReadVariableOp:cond_1/Adam/conv2d_transpose_11/bias/v/Read/ReadVariableOp2cond_1/Adam/conv2d_58/kernel/v/Read/ReadVariableOp0cond_1/Adam/conv2d_58/bias/v/Read/ReadVariableOp>cond_1/Adam/batch_normalization_55/gamma/v/Read/ReadVariableOp=cond_1/Adam/batch_normalization_55/beta/v/Read/ReadVariableOp2cond_1/Adam/conv2d_59/kernel/v/Read/ReadVariableOp0cond_1/Adam/conv2d_59/bias/v/Read/ReadVariableOpConst*?
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
__inference__traced_save_81588
?$
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_46/kernelconv2d_46/biasbatch_normalization_43/gammabatch_normalization_43/beta"batch_normalization_43/moving_mean&batch_normalization_43/moving_varianceconv2d_48/kernelconv2d_48/biasbatch_normalization_45/gammabatch_normalization_45/beta"batch_normalization_45/moving_mean&batch_normalization_45/moving_varianceconv2d_50/kernelconv2d_50/biasbatch_normalization_47/gammabatch_normalization_47/beta"batch_normalization_47/moving_mean&batch_normalization_47/moving_varianceconv2d_52/kernelconv2d_52/biasbatch_normalization_49/gammabatch_normalization_49/beta"batch_normalization_49/moving_mean&batch_normalization_49/moving_varianceconv2d_transpose_9/kernelconv2d_transpose_9/biasconv2d_54/kernelconv2d_54/biasbatch_normalization_51/gammabatch_normalization_51/beta"batch_normalization_51/moving_mean&batch_normalization_51/moving_varianceconv2d_transpose_10/kernelconv2d_transpose_10/biasconv2d_56/kernelconv2d_56/biasbatch_normalization_53/gammabatch_normalization_53/beta"batch_normalization_53/moving_mean&batch_normalization_53/moving_varianceconv2d_transpose_11/kernelconv2d_transpose_11/biasconv2d_58/kernelconv2d_58/biasbatch_normalization_55/gammabatch_normalization_55/beta"batch_normalization_55/moving_mean&batch_normalization_55/moving_varianceconv2d_59/kernelconv2d_59/biascond_1/Adam/itercond_1/Adam/beta_1cond_1/Adam/beta_2cond_1/Adam/decaycond_1/Adam/learning_ratecurrent_loss_scale
good_stepstotalcounttotal_1count_1cond_1/Adam/conv2d_46/kernel/mcond_1/Adam/conv2d_46/bias/m*cond_1/Adam/batch_normalization_43/gamma/m)cond_1/Adam/batch_normalization_43/beta/mcond_1/Adam/conv2d_48/kernel/mcond_1/Adam/conv2d_48/bias/m*cond_1/Adam/batch_normalization_45/gamma/m)cond_1/Adam/batch_normalization_45/beta/mcond_1/Adam/conv2d_50/kernel/mcond_1/Adam/conv2d_50/bias/m*cond_1/Adam/batch_normalization_47/gamma/m)cond_1/Adam/batch_normalization_47/beta/mcond_1/Adam/conv2d_52/kernel/mcond_1/Adam/conv2d_52/bias/m*cond_1/Adam/batch_normalization_49/gamma/m)cond_1/Adam/batch_normalization_49/beta/m'cond_1/Adam/conv2d_transpose_9/kernel/m%cond_1/Adam/conv2d_transpose_9/bias/mcond_1/Adam/conv2d_54/kernel/mcond_1/Adam/conv2d_54/bias/m*cond_1/Adam/batch_normalization_51/gamma/m)cond_1/Adam/batch_normalization_51/beta/m(cond_1/Adam/conv2d_transpose_10/kernel/m&cond_1/Adam/conv2d_transpose_10/bias/mcond_1/Adam/conv2d_56/kernel/mcond_1/Adam/conv2d_56/bias/m*cond_1/Adam/batch_normalization_53/gamma/m)cond_1/Adam/batch_normalization_53/beta/m(cond_1/Adam/conv2d_transpose_11/kernel/m&cond_1/Adam/conv2d_transpose_11/bias/mcond_1/Adam/conv2d_58/kernel/mcond_1/Adam/conv2d_58/bias/m*cond_1/Adam/batch_normalization_55/gamma/m)cond_1/Adam/batch_normalization_55/beta/mcond_1/Adam/conv2d_59/kernel/mcond_1/Adam/conv2d_59/bias/mcond_1/Adam/conv2d_46/kernel/vcond_1/Adam/conv2d_46/bias/v*cond_1/Adam/batch_normalization_43/gamma/v)cond_1/Adam/batch_normalization_43/beta/vcond_1/Adam/conv2d_48/kernel/vcond_1/Adam/conv2d_48/bias/v*cond_1/Adam/batch_normalization_45/gamma/v)cond_1/Adam/batch_normalization_45/beta/vcond_1/Adam/conv2d_50/kernel/vcond_1/Adam/conv2d_50/bias/v*cond_1/Adam/batch_normalization_47/gamma/v)cond_1/Adam/batch_normalization_47/beta/vcond_1/Adam/conv2d_52/kernel/vcond_1/Adam/conv2d_52/bias/v*cond_1/Adam/batch_normalization_49/gamma/v)cond_1/Adam/batch_normalization_49/beta/v'cond_1/Adam/conv2d_transpose_9/kernel/v%cond_1/Adam/conv2d_transpose_9/bias/vcond_1/Adam/conv2d_54/kernel/vcond_1/Adam/conv2d_54/bias/v*cond_1/Adam/batch_normalization_51/gamma/v)cond_1/Adam/batch_normalization_51/beta/v(cond_1/Adam/conv2d_transpose_10/kernel/v&cond_1/Adam/conv2d_transpose_10/bias/vcond_1/Adam/conv2d_56/kernel/vcond_1/Adam/conv2d_56/bias/v*cond_1/Adam/batch_normalization_53/gamma/v)cond_1/Adam/batch_normalization_53/beta/v(cond_1/Adam/conv2d_transpose_11/kernel/v&cond_1/Adam/conv2d_transpose_11/bias/vcond_1/Adam/conv2d_58/kernel/vcond_1/Adam/conv2d_58/bias/v*cond_1/Adam/batch_normalization_55/gamma/v)cond_1/Adam/batch_normalization_55/beta/vcond_1/Adam/conv2d_59/kernel/vcond_1/Adam/conv2d_59/bias/v*?
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
!__inference__traced_restore_81997??'
?
d
H__inference_activation_45_layer_call_and_return_conditional_losses_77483

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
ñ
?
B__inference_model_3_layer_call_and_return_conditional_losses_78448
input_4
conv2d_46_78310
conv2d_46_78312 
batch_normalization_43_78315 
batch_normalization_43_78317 
batch_normalization_43_78319 
batch_normalization_43_78321
conv2d_48_78327
conv2d_48_78329 
batch_normalization_45_78332 
batch_normalization_45_78334 
batch_normalization_45_78336 
batch_normalization_45_78338
conv2d_50_78344
conv2d_50_78346 
batch_normalization_47_78349 
batch_normalization_47_78351 
batch_normalization_47_78353 
batch_normalization_47_78355
conv2d_52_78361
conv2d_52_78363 
batch_normalization_49_78366 
batch_normalization_49_78368 
batch_normalization_49_78370 
batch_normalization_49_78372
conv2d_transpose_9_78376
conv2d_transpose_9_78378
conv2d_54_78383
conv2d_54_78385 
batch_normalization_51_78388 
batch_normalization_51_78390 
batch_normalization_51_78392 
batch_normalization_51_78394
conv2d_transpose_10_78398
conv2d_transpose_10_78400
conv2d_56_78405
conv2d_56_78407 
batch_normalization_53_78410 
batch_normalization_53_78412 
batch_normalization_53_78414 
batch_normalization_53_78416
conv2d_transpose_11_78420
conv2d_transpose_11_78422
conv2d_58_78427
conv2d_58_78429 
batch_normalization_55_78432 
batch_normalization_55_78434 
batch_normalization_55_78436 
batch_normalization_55_78438
conv2d_59_78442
conv2d_59_78444
identity??.batch_normalization_43/StatefulPartitionedCall?.batch_normalization_45/StatefulPartitionedCall?.batch_normalization_47/StatefulPartitionedCall?.batch_normalization_49/StatefulPartitionedCall?.batch_normalization_51/StatefulPartitionedCall?.batch_normalization_53/StatefulPartitionedCall?.batch_normalization_55/StatefulPartitionedCall?!conv2d_46/StatefulPartitionedCall?!conv2d_48/StatefulPartitionedCall?!conv2d_50/StatefulPartitionedCall?!conv2d_52/StatefulPartitionedCall?!conv2d_54/StatefulPartitionedCall?!conv2d_56/StatefulPartitionedCall?!conv2d_58/StatefulPartitionedCall?!conv2d_59/StatefulPartitionedCall?+conv2d_transpose_10/StatefulPartitionedCall?+conv2d_transpose_11/StatefulPartitionedCall?*conv2d_transpose_9/StatefulPartitionedCallh
CastCastinput_4*

DstT0*

SrcT0*1
_output_shapes
:???????????2
Cast?
!conv2d_46/StatefulPartitionedCallStatefulPartitionedCallCast:y:0conv2d_46_78310conv2d_46_78312*
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
D__inference_conv2d_46_layer_call_and_return_conditional_losses_772442#
!conv2d_46/StatefulPartitionedCall?
.batch_normalization_43/StatefulPartitionedCallStatefulPartitionedCall*conv2d_46/StatefulPartitionedCall:output:0batch_normalization_43_78315batch_normalization_43_78317batch_normalization_43_78319batch_normalization_43_78321*
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
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_7729720
.batch_normalization_43/StatefulPartitionedCall?
activation_43/PartitionedCallPartitionedCall7batch_normalization_43/StatefulPartitionedCall:output:0*
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
H__inference_activation_43_layer_call_and_return_conditional_losses_773382
activation_43/PartitionedCall?
max_pooling2d_9/PartitionedCallPartitionedCall&activation_43/PartitionedCall:output:0*
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
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_764352!
max_pooling2d_9/PartitionedCall?
dropout_18/PartitionedCallPartitionedCall(max_pooling2d_9/PartitionedCall:output:0*
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
  ?E8? *N
fIRG
E__inference_dropout_18_layer_call_and_return_conditional_losses_773642
dropout_18/PartitionedCall?
!conv2d_48/StatefulPartitionedCallStatefulPartitionedCall#dropout_18/PartitionedCall:output:0conv2d_48_78327conv2d_48_78329*
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
D__inference_conv2d_48_layer_call_and_return_conditional_losses_773892#
!conv2d_48/StatefulPartitionedCall?
.batch_normalization_45/StatefulPartitionedCallStatefulPartitionedCall*conv2d_48/StatefulPartitionedCall:output:0batch_normalization_45_78332batch_normalization_45_78334batch_normalization_45_78336batch_normalization_45_78338*
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
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_7744220
.batch_normalization_45/StatefulPartitionedCall?
activation_45/PartitionedCallPartitionedCall7batch_normalization_45/StatefulPartitionedCall:output:0*
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
H__inference_activation_45_layer_call_and_return_conditional_losses_774832
activation_45/PartitionedCall?
 max_pooling2d_10/PartitionedCallPartitionedCall&activation_45/PartitionedCall:output:0*
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
  ?E8? *T
fORM
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_765512"
 max_pooling2d_10/PartitionedCall?
dropout_19/PartitionedCallPartitionedCall)max_pooling2d_10/PartitionedCall:output:0*
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
  ?E8? *N
fIRG
E__inference_dropout_19_layer_call_and_return_conditional_losses_775092
dropout_19/PartitionedCall?
!conv2d_50/StatefulPartitionedCallStatefulPartitionedCall#dropout_19/PartitionedCall:output:0conv2d_50_78344conv2d_50_78346*
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
D__inference_conv2d_50_layer_call_and_return_conditional_losses_775342#
!conv2d_50/StatefulPartitionedCall?
.batch_normalization_47/StatefulPartitionedCallStatefulPartitionedCall*conv2d_50/StatefulPartitionedCall:output:0batch_normalization_47_78349batch_normalization_47_78351batch_normalization_47_78353batch_normalization_47_78355*
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
Q__inference_batch_normalization_47_layer_call_and_return_conditional_losses_7758720
.batch_normalization_47/StatefulPartitionedCall?
activation_47/PartitionedCallPartitionedCall7batch_normalization_47/StatefulPartitionedCall:output:0*
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
H__inference_activation_47_layer_call_and_return_conditional_losses_776282
activation_47/PartitionedCall?
 max_pooling2d_11/PartitionedCallPartitionedCall&activation_47/PartitionedCall:output:0*
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
  ?E8? *T
fORM
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_766672"
 max_pooling2d_11/PartitionedCall?
dropout_20/PartitionedCallPartitionedCall)max_pooling2d_11/PartitionedCall:output:0*
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
  ?E8? *N
fIRG
E__inference_dropout_20_layer_call_and_return_conditional_losses_776542
dropout_20/PartitionedCall?
!conv2d_52/StatefulPartitionedCallStatefulPartitionedCall#dropout_20/PartitionedCall:output:0conv2d_52_78361conv2d_52_78363*
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
D__inference_conv2d_52_layer_call_and_return_conditional_losses_776792#
!conv2d_52/StatefulPartitionedCall?
.batch_normalization_49/StatefulPartitionedCallStatefulPartitionedCall*conv2d_52/StatefulPartitionedCall:output:0batch_normalization_49_78366batch_normalization_49_78368batch_normalization_49_78370batch_normalization_49_78372*
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
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_7773220
.batch_normalization_49/StatefulPartitionedCall?
activation_49/PartitionedCallPartitionedCall7batch_normalization_49/StatefulPartitionedCall:output:0*
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
H__inference_activation_49_layer_call_and_return_conditional_losses_777732
activation_49/PartitionedCall?
*conv2d_transpose_9/StatefulPartitionedCallStatefulPartitionedCall&activation_49/PartitionedCall:output:0conv2d_transpose_9_78376conv2d_transpose_9_78378*
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
M__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_768132,
*conv2d_transpose_9/StatefulPartitionedCall?
concatenate_9/PartitionedCallPartitionedCall3conv2d_transpose_9/StatefulPartitionedCall:output:0&activation_47/PartitionedCall:output:0*
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
H__inference_concatenate_9_layer_call_and_return_conditional_losses_777932
concatenate_9/PartitionedCall?
dropout_21/PartitionedCallPartitionedCall&concatenate_9/PartitionedCall:output:0*
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
  ?E8? *N
fIRG
E__inference_dropout_21_layer_call_and_return_conditional_losses_778192
dropout_21/PartitionedCall?
!conv2d_54/StatefulPartitionedCallStatefulPartitionedCall#dropout_21/PartitionedCall:output:0conv2d_54_78383conv2d_54_78385*
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
D__inference_conv2d_54_layer_call_and_return_conditional_losses_778442#
!conv2d_54/StatefulPartitionedCall?
.batch_normalization_51/StatefulPartitionedCallStatefulPartitionedCall*conv2d_54/StatefulPartitionedCall:output:0batch_normalization_51_78388batch_normalization_51_78390batch_normalization_51_78392batch_normalization_51_78394*
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
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_7789720
.batch_normalization_51/StatefulPartitionedCall?
activation_51/PartitionedCallPartitionedCall7batch_normalization_51/StatefulPartitionedCall:output:0*
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
H__inference_activation_51_layer_call_and_return_conditional_losses_779382
activation_51/PartitionedCall?
+conv2d_transpose_10/StatefulPartitionedCallStatefulPartitionedCall&activation_51/PartitionedCall:output:0conv2d_transpose_10_78398conv2d_transpose_10_78400*
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
  ?E8? *W
fRRP
N__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_769632-
+conv2d_transpose_10/StatefulPartitionedCall?
concatenate_10/PartitionedCallPartitionedCall4conv2d_transpose_10/StatefulPartitionedCall:output:0&activation_45/PartitionedCall:output:0*
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
  ?E8? *R
fMRK
I__inference_concatenate_10_layer_call_and_return_conditional_losses_779582 
concatenate_10/PartitionedCall?
dropout_22/PartitionedCallPartitionedCall'concatenate_10/PartitionedCall:output:0*
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
E__inference_dropout_22_layer_call_and_return_conditional_losses_779842
dropout_22/PartitionedCall?
!conv2d_56/StatefulPartitionedCallStatefulPartitionedCall#dropout_22/PartitionedCall:output:0conv2d_56_78405conv2d_56_78407*
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
D__inference_conv2d_56_layer_call_and_return_conditional_losses_780092#
!conv2d_56/StatefulPartitionedCall?
.batch_normalization_53/StatefulPartitionedCallStatefulPartitionedCall*conv2d_56/StatefulPartitionedCall:output:0batch_normalization_53_78410batch_normalization_53_78412batch_normalization_53_78414batch_normalization_53_78416*
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
Q__inference_batch_normalization_53_layer_call_and_return_conditional_losses_7806220
.batch_normalization_53/StatefulPartitionedCall?
activation_53/PartitionedCallPartitionedCall7batch_normalization_53/StatefulPartitionedCall:output:0*
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
H__inference_activation_53_layer_call_and_return_conditional_losses_781032
activation_53/PartitionedCall?
+conv2d_transpose_11/StatefulPartitionedCallStatefulPartitionedCall&activation_53/PartitionedCall:output:0conv2d_transpose_11_78420conv2d_transpose_11_78422*
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
  ?E8? *W
fRRP
N__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_771132-
+conv2d_transpose_11/StatefulPartitionedCall?
concatenate_11/PartitionedCallPartitionedCall4conv2d_transpose_11/StatefulPartitionedCall:output:0&activation_43/PartitionedCall:output:0*
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
  ?E8? *R
fMRK
I__inference_concatenate_11_layer_call_and_return_conditional_losses_781232 
concatenate_11/PartitionedCall?
dropout_23/PartitionedCallPartitionedCall'concatenate_11/PartitionedCall:output:0*
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
E__inference_dropout_23_layer_call_and_return_conditional_losses_781492
dropout_23/PartitionedCall?
!conv2d_58/StatefulPartitionedCallStatefulPartitionedCall#dropout_23/PartitionedCall:output:0conv2d_58_78427conv2d_58_78429*
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
D__inference_conv2d_58_layer_call_and_return_conditional_losses_781742#
!conv2d_58/StatefulPartitionedCall?
.batch_normalization_55/StatefulPartitionedCallStatefulPartitionedCall*conv2d_58/StatefulPartitionedCall:output:0batch_normalization_55_78432batch_normalization_55_78434batch_normalization_55_78436batch_normalization_55_78438*
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
Q__inference_batch_normalization_55_layer_call_and_return_conditional_losses_7822720
.batch_normalization_55/StatefulPartitionedCall?
activation_55/PartitionedCallPartitionedCall7batch_normalization_55/StatefulPartitionedCall:output:0*
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
H__inference_activation_55_layer_call_and_return_conditional_losses_782682
activation_55/PartitionedCall?
!conv2d_59/StatefulPartitionedCallStatefulPartitionedCall&activation_55/PartitionedCall:output:0conv2d_59_78442conv2d_59_78444*
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
D__inference_conv2d_59_layer_call_and_return_conditional_losses_782892#
!conv2d_59/StatefulPartitionedCall?
IdentityIdentity*conv2d_59/StatefulPartitionedCall:output:0/^batch_normalization_43/StatefulPartitionedCall/^batch_normalization_45/StatefulPartitionedCall/^batch_normalization_47/StatefulPartitionedCall/^batch_normalization_49/StatefulPartitionedCall/^batch_normalization_51/StatefulPartitionedCall/^batch_normalization_53/StatefulPartitionedCall/^batch_normalization_55/StatefulPartitionedCall"^conv2d_46/StatefulPartitionedCall"^conv2d_48/StatefulPartitionedCall"^conv2d_50/StatefulPartitionedCall"^conv2d_52/StatefulPartitionedCall"^conv2d_54/StatefulPartitionedCall"^conv2d_56/StatefulPartitionedCall"^conv2d_58/StatefulPartitionedCall"^conv2d_59/StatefulPartitionedCall,^conv2d_transpose_10/StatefulPartitionedCall,^conv2d_transpose_11/StatefulPartitionedCall+^conv2d_transpose_9/StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_43/StatefulPartitionedCall.batch_normalization_43/StatefulPartitionedCall2`
.batch_normalization_45/StatefulPartitionedCall.batch_normalization_45/StatefulPartitionedCall2`
.batch_normalization_47/StatefulPartitionedCall.batch_normalization_47/StatefulPartitionedCall2`
.batch_normalization_49/StatefulPartitionedCall.batch_normalization_49/StatefulPartitionedCall2`
.batch_normalization_51/StatefulPartitionedCall.batch_normalization_51/StatefulPartitionedCall2`
.batch_normalization_53/StatefulPartitionedCall.batch_normalization_53/StatefulPartitionedCall2`
.batch_normalization_55/StatefulPartitionedCall.batch_normalization_55/StatefulPartitionedCall2F
!conv2d_46/StatefulPartitionedCall!conv2d_46/StatefulPartitionedCall2F
!conv2d_48/StatefulPartitionedCall!conv2d_48/StatefulPartitionedCall2F
!conv2d_50/StatefulPartitionedCall!conv2d_50/StatefulPartitionedCall2F
!conv2d_52/StatefulPartitionedCall!conv2d_52/StatefulPartitionedCall2F
!conv2d_54/StatefulPartitionedCall!conv2d_54/StatefulPartitionedCall2F
!conv2d_56/StatefulPartitionedCall!conv2d_56/StatefulPartitionedCall2F
!conv2d_58/StatefulPartitionedCall!conv2d_58/StatefulPartitionedCall2F
!conv2d_59/StatefulPartitionedCall!conv2d_59/StatefulPartitionedCall2Z
+conv2d_transpose_10/StatefulPartitionedCall+conv2d_transpose_10/StatefulPartitionedCall2Z
+conv2d_transpose_11/StatefulPartitionedCall+conv2d_transpose_11/StatefulPartitionedCall2X
*conv2d_transpose_9/StatefulPartitionedCall*conv2d_transpose_9/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_4
?
?
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_80692

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
2__inference_conv2d_transpose_9_layer_call_fn_76823

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
M__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_768132
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
d
H__inference_activation_47_layer_call_and_return_conditional_losses_77628

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
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_80646

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
?
?
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_77279

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
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_79871

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
F
*__inference_dropout_20_layer_call_fn_80388

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
  ?E8? *N
fIRG
E__inference_dropout_20_layer_call_and_return_conditional_losses_776542
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
Q__inference_batch_normalization_55_layer_call_and_return_conditional_losses_78209

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
?
?
'__inference_model_3_layer_call_fn_79725

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
B__inference_model_3_layer_call_and_return_conditional_losses_785932
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
?
?
6__inference_batch_normalization_55_layer_call_fn_81134

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
Q__inference_batch_normalization_55_layer_call_and_return_conditional_losses_772162
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
?
?
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_77714

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
?
?
6__inference_batch_normalization_49_layer_call_fn_80460

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
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_777142
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
?
?
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_76916

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
?
Y
-__inference_concatenate_9_layer_call_fn_80560
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
H__inference_concatenate_9_layer_call_and_return_conditional_losses_777932
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
?
I
-__inference_activation_49_layer_call_fn_80547

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
H__inference_activation_49_layer_call_and_return_conditional_losses_777732
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
6__inference_batch_normalization_51_layer_call_fn_80736

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
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_778972
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
Q__inference_batch_normalization_53_layer_call_and_return_conditional_losses_80827

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
?
?
D__inference_conv2d_54_layer_call_and_return_conditional_losses_77844

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
?
?
6__inference_batch_normalization_43_layer_call_fn_79966

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
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_772792
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
Q__inference_batch_normalization_53_layer_call_and_return_conditional_losses_78044

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
?
?
D__inference_conv2d_58_layer_call_and_return_conditional_losses_80997

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
?
?
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_76534

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
?
?
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_77879

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
?
?
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_80057

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
Q__inference_batch_normalization_47_layer_call_and_return_conditional_losses_80243

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
Z
.__inference_concatenate_10_layer_call_fn_80759
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
  ?E8? *R
fMRK
I__inference_concatenate_10_layer_call_and_return_conditional_losses_779582
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
?
?
Q__inference_batch_normalization_47_layer_call_and_return_conditional_losses_80325

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
?
?
Q__inference_batch_normalization_55_layer_call_and_return_conditional_losses_81026

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
?
c
E__inference_dropout_20_layer_call_and_return_conditional_losses_77654

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
?
?
D__inference_conv2d_54_layer_call_and_return_conditional_losses_80599

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
c
*__inference_dropout_23_layer_call_fn_80980

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
E__inference_dropout_23_layer_call_and_return_conditional_losses_781442
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
?
d
E__inference_dropout_18_layer_call_and_return_conditional_losses_77359

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
?
s
I__inference_concatenate_11_layer_call_and_return_conditional_losses_78123

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
?
?
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_76387

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
?
c
*__inference_dropout_22_layer_call_fn_80781

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
E__inference_dropout_22_layer_call_and_return_conditional_losses_779792
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
?
c
*__inference_dropout_20_layer_call_fn_80383

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
  ?E8? *N
fIRG
E__inference_dropout_20_layer_call_and_return_conditional_losses_776492
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
?
?
6__inference_batch_normalization_47_layer_call_fn_80351

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
Q__inference_batch_normalization_47_layer_call_and_return_conditional_losses_766502
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
?
?
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_77297

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
?
?
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_80429

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
?
?
D__inference_conv2d_59_layer_call_and_return_conditional_losses_81157

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
?
?
6__inference_batch_normalization_47_layer_call_fn_80338

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
Q__inference_batch_normalization_47_layer_call_and_return_conditional_losses_766192
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
??
?Q
!__inference__traced_restore_81997
file_prefix%
!assignvariableop_conv2d_46_kernel%
!assignvariableop_1_conv2d_46_bias3
/assignvariableop_2_batch_normalization_43_gamma2
.assignvariableop_3_batch_normalization_43_beta9
5assignvariableop_4_batch_normalization_43_moving_mean=
9assignvariableop_5_batch_normalization_43_moving_variance'
#assignvariableop_6_conv2d_48_kernel%
!assignvariableop_7_conv2d_48_bias3
/assignvariableop_8_batch_normalization_45_gamma2
.assignvariableop_9_batch_normalization_45_beta:
6assignvariableop_10_batch_normalization_45_moving_mean>
:assignvariableop_11_batch_normalization_45_moving_variance(
$assignvariableop_12_conv2d_50_kernel&
"assignvariableop_13_conv2d_50_bias4
0assignvariableop_14_batch_normalization_47_gamma3
/assignvariableop_15_batch_normalization_47_beta:
6assignvariableop_16_batch_normalization_47_moving_mean>
:assignvariableop_17_batch_normalization_47_moving_variance(
$assignvariableop_18_conv2d_52_kernel&
"assignvariableop_19_conv2d_52_bias4
0assignvariableop_20_batch_normalization_49_gamma3
/assignvariableop_21_batch_normalization_49_beta:
6assignvariableop_22_batch_normalization_49_moving_mean>
:assignvariableop_23_batch_normalization_49_moving_variance1
-assignvariableop_24_conv2d_transpose_9_kernel/
+assignvariableop_25_conv2d_transpose_9_bias(
$assignvariableop_26_conv2d_54_kernel&
"assignvariableop_27_conv2d_54_bias4
0assignvariableop_28_batch_normalization_51_gamma3
/assignvariableop_29_batch_normalization_51_beta:
6assignvariableop_30_batch_normalization_51_moving_mean>
:assignvariableop_31_batch_normalization_51_moving_variance2
.assignvariableop_32_conv2d_transpose_10_kernel0
,assignvariableop_33_conv2d_transpose_10_bias(
$assignvariableop_34_conv2d_56_kernel&
"assignvariableop_35_conv2d_56_bias4
0assignvariableop_36_batch_normalization_53_gamma3
/assignvariableop_37_batch_normalization_53_beta:
6assignvariableop_38_batch_normalization_53_moving_mean>
:assignvariableop_39_batch_normalization_53_moving_variance2
.assignvariableop_40_conv2d_transpose_11_kernel0
,assignvariableop_41_conv2d_transpose_11_bias(
$assignvariableop_42_conv2d_58_kernel&
"assignvariableop_43_conv2d_58_bias4
0assignvariableop_44_batch_normalization_55_gamma3
/assignvariableop_45_batch_normalization_55_beta:
6assignvariableop_46_batch_normalization_55_moving_mean>
:assignvariableop_47_batch_normalization_55_moving_variance(
$assignvariableop_48_conv2d_59_kernel&
"assignvariableop_49_conv2d_59_bias(
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
2assignvariableop_61_cond_1_adam_conv2d_46_kernel_m4
0assignvariableop_62_cond_1_adam_conv2d_46_bias_mB
>assignvariableop_63_cond_1_adam_batch_normalization_43_gamma_mA
=assignvariableop_64_cond_1_adam_batch_normalization_43_beta_m6
2assignvariableop_65_cond_1_adam_conv2d_48_kernel_m4
0assignvariableop_66_cond_1_adam_conv2d_48_bias_mB
>assignvariableop_67_cond_1_adam_batch_normalization_45_gamma_mA
=assignvariableop_68_cond_1_adam_batch_normalization_45_beta_m6
2assignvariableop_69_cond_1_adam_conv2d_50_kernel_m4
0assignvariableop_70_cond_1_adam_conv2d_50_bias_mB
>assignvariableop_71_cond_1_adam_batch_normalization_47_gamma_mA
=assignvariableop_72_cond_1_adam_batch_normalization_47_beta_m6
2assignvariableop_73_cond_1_adam_conv2d_52_kernel_m4
0assignvariableop_74_cond_1_adam_conv2d_52_bias_mB
>assignvariableop_75_cond_1_adam_batch_normalization_49_gamma_mA
=assignvariableop_76_cond_1_adam_batch_normalization_49_beta_m?
;assignvariableop_77_cond_1_adam_conv2d_transpose_9_kernel_m=
9assignvariableop_78_cond_1_adam_conv2d_transpose_9_bias_m6
2assignvariableop_79_cond_1_adam_conv2d_54_kernel_m4
0assignvariableop_80_cond_1_adam_conv2d_54_bias_mB
>assignvariableop_81_cond_1_adam_batch_normalization_51_gamma_mA
=assignvariableop_82_cond_1_adam_batch_normalization_51_beta_m@
<assignvariableop_83_cond_1_adam_conv2d_transpose_10_kernel_m>
:assignvariableop_84_cond_1_adam_conv2d_transpose_10_bias_m6
2assignvariableop_85_cond_1_adam_conv2d_56_kernel_m4
0assignvariableop_86_cond_1_adam_conv2d_56_bias_mB
>assignvariableop_87_cond_1_adam_batch_normalization_53_gamma_mA
=assignvariableop_88_cond_1_adam_batch_normalization_53_beta_m@
<assignvariableop_89_cond_1_adam_conv2d_transpose_11_kernel_m>
:assignvariableop_90_cond_1_adam_conv2d_transpose_11_bias_m6
2assignvariableop_91_cond_1_adam_conv2d_58_kernel_m4
0assignvariableop_92_cond_1_adam_conv2d_58_bias_mB
>assignvariableop_93_cond_1_adam_batch_normalization_55_gamma_mA
=assignvariableop_94_cond_1_adam_batch_normalization_55_beta_m6
2assignvariableop_95_cond_1_adam_conv2d_59_kernel_m4
0assignvariableop_96_cond_1_adam_conv2d_59_bias_m6
2assignvariableop_97_cond_1_adam_conv2d_46_kernel_v4
0assignvariableop_98_cond_1_adam_conv2d_46_bias_vB
>assignvariableop_99_cond_1_adam_batch_normalization_43_gamma_vB
>assignvariableop_100_cond_1_adam_batch_normalization_43_beta_v7
3assignvariableop_101_cond_1_adam_conv2d_48_kernel_v5
1assignvariableop_102_cond_1_adam_conv2d_48_bias_vC
?assignvariableop_103_cond_1_adam_batch_normalization_45_gamma_vB
>assignvariableop_104_cond_1_adam_batch_normalization_45_beta_v7
3assignvariableop_105_cond_1_adam_conv2d_50_kernel_v5
1assignvariableop_106_cond_1_adam_conv2d_50_bias_vC
?assignvariableop_107_cond_1_adam_batch_normalization_47_gamma_vB
>assignvariableop_108_cond_1_adam_batch_normalization_47_beta_v7
3assignvariableop_109_cond_1_adam_conv2d_52_kernel_v5
1assignvariableop_110_cond_1_adam_conv2d_52_bias_vC
?assignvariableop_111_cond_1_adam_batch_normalization_49_gamma_vB
>assignvariableop_112_cond_1_adam_batch_normalization_49_beta_v@
<assignvariableop_113_cond_1_adam_conv2d_transpose_9_kernel_v>
:assignvariableop_114_cond_1_adam_conv2d_transpose_9_bias_v7
3assignvariableop_115_cond_1_adam_conv2d_54_kernel_v5
1assignvariableop_116_cond_1_adam_conv2d_54_bias_vC
?assignvariableop_117_cond_1_adam_batch_normalization_51_gamma_vB
>assignvariableop_118_cond_1_adam_batch_normalization_51_beta_vA
=assignvariableop_119_cond_1_adam_conv2d_transpose_10_kernel_v?
;assignvariableop_120_cond_1_adam_conv2d_transpose_10_bias_v7
3assignvariableop_121_cond_1_adam_conv2d_56_kernel_v5
1assignvariableop_122_cond_1_adam_conv2d_56_bias_vC
?assignvariableop_123_cond_1_adam_batch_normalization_53_gamma_vB
>assignvariableop_124_cond_1_adam_batch_normalization_53_beta_vA
=assignvariableop_125_cond_1_adam_conv2d_transpose_11_kernel_v?
;assignvariableop_126_cond_1_adam_conv2d_transpose_11_bias_v7
3assignvariableop_127_cond_1_adam_conv2d_58_kernel_v5
1assignvariableop_128_cond_1_adam_conv2d_58_bias_vC
?assignvariableop_129_cond_1_adam_batch_normalization_55_gamma_vB
>assignvariableop_130_cond_1_adam_batch_normalization_55_beta_v7
3assignvariableop_131_cond_1_adam_conv2d_59_kernel_v5
1assignvariableop_132_cond_1_adam_conv2d_59_bias_v
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
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_46_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_46_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_43_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_43_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_43_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_43_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_48_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_48_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_45_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_45_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_45_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_45_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp$assignvariableop_12_conv2d_50_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp"assignvariableop_13_conv2d_50_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp0assignvariableop_14_batch_normalization_47_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp/assignvariableop_15_batch_normalization_47_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp6assignvariableop_16_batch_normalization_47_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp:assignvariableop_17_batch_normalization_47_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp$assignvariableop_18_conv2d_52_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp"assignvariableop_19_conv2d_52_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp0assignvariableop_20_batch_normalization_49_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp/assignvariableop_21_batch_normalization_49_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp6assignvariableop_22_batch_normalization_49_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp:assignvariableop_23_batch_normalization_49_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp-assignvariableop_24_conv2d_transpose_9_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp+assignvariableop_25_conv2d_transpose_9_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp$assignvariableop_26_conv2d_54_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp"assignvariableop_27_conv2d_54_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp0assignvariableop_28_batch_normalization_51_gammaIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp/assignvariableop_29_batch_normalization_51_betaIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp6assignvariableop_30_batch_normalization_51_moving_meanIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp:assignvariableop_31_batch_normalization_51_moving_varianceIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp.assignvariableop_32_conv2d_transpose_10_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp,assignvariableop_33_conv2d_transpose_10_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp$assignvariableop_34_conv2d_56_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp"assignvariableop_35_conv2d_56_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp0assignvariableop_36_batch_normalization_53_gammaIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp/assignvariableop_37_batch_normalization_53_betaIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp6assignvariableop_38_batch_normalization_53_moving_meanIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp:assignvariableop_39_batch_normalization_53_moving_varianceIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp.assignvariableop_40_conv2d_transpose_11_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp,assignvariableop_41_conv2d_transpose_11_biasIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp$assignvariableop_42_conv2d_58_kernelIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp"assignvariableop_43_conv2d_58_biasIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp0assignvariableop_44_batch_normalization_55_gammaIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp/assignvariableop_45_batch_normalization_55_betaIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp6assignvariableop_46_batch_normalization_55_moving_meanIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp:assignvariableop_47_batch_normalization_55_moving_varianceIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp$assignvariableop_48_conv2d_59_kernelIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp"assignvariableop_49_conv2d_59_biasIdentity_49:output:0"/device:CPU:0*
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
AssignVariableOp_61AssignVariableOp2assignvariableop_61_cond_1_adam_conv2d_46_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp0assignvariableop_62_cond_1_adam_conv2d_46_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp>assignvariableop_63_cond_1_adam_batch_normalization_43_gamma_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp=assignvariableop_64_cond_1_adam_batch_normalization_43_beta_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp2assignvariableop_65_cond_1_adam_conv2d_48_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp0assignvariableop_66_cond_1_adam_conv2d_48_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp>assignvariableop_67_cond_1_adam_batch_normalization_45_gamma_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp=assignvariableop_68_cond_1_adam_batch_normalization_45_beta_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp2assignvariableop_69_cond_1_adam_conv2d_50_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp0assignvariableop_70_cond_1_adam_conv2d_50_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp>assignvariableop_71_cond_1_adam_batch_normalization_47_gamma_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp=assignvariableop_72_cond_1_adam_batch_normalization_47_beta_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOp2assignvariableop_73_cond_1_adam_conv2d_52_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp0assignvariableop_74_cond_1_adam_conv2d_52_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOp>assignvariableop_75_cond_1_adam_batch_normalization_49_gamma_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOp=assignvariableop_76_cond_1_adam_batch_normalization_49_beta_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOp;assignvariableop_77_cond_1_adam_conv2d_transpose_9_kernel_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOp9assignvariableop_78_cond_1_adam_conv2d_transpose_9_bias_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79?
AssignVariableOp_79AssignVariableOp2assignvariableop_79_cond_1_adam_conv2d_54_kernel_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80?
AssignVariableOp_80AssignVariableOp0assignvariableop_80_cond_1_adam_conv2d_54_bias_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81?
AssignVariableOp_81AssignVariableOp>assignvariableop_81_cond_1_adam_batch_normalization_51_gamma_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82?
AssignVariableOp_82AssignVariableOp=assignvariableop_82_cond_1_adam_batch_normalization_51_beta_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83?
AssignVariableOp_83AssignVariableOp<assignvariableop_83_cond_1_adam_conv2d_transpose_10_kernel_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84?
AssignVariableOp_84AssignVariableOp:assignvariableop_84_cond_1_adam_conv2d_transpose_10_bias_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85?
AssignVariableOp_85AssignVariableOp2assignvariableop_85_cond_1_adam_conv2d_56_kernel_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86?
AssignVariableOp_86AssignVariableOp0assignvariableop_86_cond_1_adam_conv2d_56_bias_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87?
AssignVariableOp_87AssignVariableOp>assignvariableop_87_cond_1_adam_batch_normalization_53_gamma_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88?
AssignVariableOp_88AssignVariableOp=assignvariableop_88_cond_1_adam_batch_normalization_53_beta_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89?
AssignVariableOp_89AssignVariableOp<assignvariableop_89_cond_1_adam_conv2d_transpose_11_kernel_mIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90?
AssignVariableOp_90AssignVariableOp:assignvariableop_90_cond_1_adam_conv2d_transpose_11_bias_mIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91?
AssignVariableOp_91AssignVariableOp2assignvariableop_91_cond_1_adam_conv2d_58_kernel_mIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92?
AssignVariableOp_92AssignVariableOp0assignvariableop_92_cond_1_adam_conv2d_58_bias_mIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93?
AssignVariableOp_93AssignVariableOp>assignvariableop_93_cond_1_adam_batch_normalization_55_gamma_mIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94?
AssignVariableOp_94AssignVariableOp=assignvariableop_94_cond_1_adam_batch_normalization_55_beta_mIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95?
AssignVariableOp_95AssignVariableOp2assignvariableop_95_cond_1_adam_conv2d_59_kernel_mIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96?
AssignVariableOp_96AssignVariableOp0assignvariableop_96_cond_1_adam_conv2d_59_bias_mIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97?
AssignVariableOp_97AssignVariableOp2assignvariableop_97_cond_1_adam_conv2d_46_kernel_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98?
AssignVariableOp_98AssignVariableOp0assignvariableop_98_cond_1_adam_conv2d_46_bias_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_98n
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:2
Identity_99?
AssignVariableOp_99AssignVariableOp>assignvariableop_99_cond_1_adam_batch_normalization_43_gamma_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99q
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:2
Identity_100?
AssignVariableOp_100AssignVariableOp>assignvariableop_100_cond_1_adam_batch_normalization_43_beta_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_100q
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:2
Identity_101?
AssignVariableOp_101AssignVariableOp3assignvariableop_101_cond_1_adam_conv2d_48_kernel_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_101q
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:2
Identity_102?
AssignVariableOp_102AssignVariableOp1assignvariableop_102_cond_1_adam_conv2d_48_bias_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_102q
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:2
Identity_103?
AssignVariableOp_103AssignVariableOp?assignvariableop_103_cond_1_adam_batch_normalization_45_gamma_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_103q
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:2
Identity_104?
AssignVariableOp_104AssignVariableOp>assignvariableop_104_cond_1_adam_batch_normalization_45_beta_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_104q
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:2
Identity_105?
AssignVariableOp_105AssignVariableOp3assignvariableop_105_cond_1_adam_conv2d_50_kernel_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_105q
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:2
Identity_106?
AssignVariableOp_106AssignVariableOp1assignvariableop_106_cond_1_adam_conv2d_50_bias_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_106q
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:2
Identity_107?
AssignVariableOp_107AssignVariableOp?assignvariableop_107_cond_1_adam_batch_normalization_47_gamma_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_107q
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:2
Identity_108?
AssignVariableOp_108AssignVariableOp>assignvariableop_108_cond_1_adam_batch_normalization_47_beta_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_108q
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:2
Identity_109?
AssignVariableOp_109AssignVariableOp3assignvariableop_109_cond_1_adam_conv2d_52_kernel_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_109q
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:2
Identity_110?
AssignVariableOp_110AssignVariableOp1assignvariableop_110_cond_1_adam_conv2d_52_bias_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_110q
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:2
Identity_111?
AssignVariableOp_111AssignVariableOp?assignvariableop_111_cond_1_adam_batch_normalization_49_gamma_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_111q
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:2
Identity_112?
AssignVariableOp_112AssignVariableOp>assignvariableop_112_cond_1_adam_batch_normalization_49_beta_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_112q
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:2
Identity_113?
AssignVariableOp_113AssignVariableOp<assignvariableop_113_cond_1_adam_conv2d_transpose_9_kernel_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_113q
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:2
Identity_114?
AssignVariableOp_114AssignVariableOp:assignvariableop_114_cond_1_adam_conv2d_transpose_9_bias_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_114q
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:2
Identity_115?
AssignVariableOp_115AssignVariableOp3assignvariableop_115_cond_1_adam_conv2d_54_kernel_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_115q
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:2
Identity_116?
AssignVariableOp_116AssignVariableOp1assignvariableop_116_cond_1_adam_conv2d_54_bias_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_116q
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:2
Identity_117?
AssignVariableOp_117AssignVariableOp?assignvariableop_117_cond_1_adam_batch_normalization_51_gamma_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_117q
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:2
Identity_118?
AssignVariableOp_118AssignVariableOp>assignvariableop_118_cond_1_adam_batch_normalization_51_beta_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_118q
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:2
Identity_119?
AssignVariableOp_119AssignVariableOp=assignvariableop_119_cond_1_adam_conv2d_transpose_10_kernel_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119q
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:2
Identity_120?
AssignVariableOp_120AssignVariableOp;assignvariableop_120_cond_1_adam_conv2d_transpose_10_bias_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_120q
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:2
Identity_121?
AssignVariableOp_121AssignVariableOp3assignvariableop_121_cond_1_adam_conv2d_56_kernel_vIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_121q
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:2
Identity_122?
AssignVariableOp_122AssignVariableOp1assignvariableop_122_cond_1_adam_conv2d_56_bias_vIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_122q
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:2
Identity_123?
AssignVariableOp_123AssignVariableOp?assignvariableop_123_cond_1_adam_batch_normalization_53_gamma_vIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_123q
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:2
Identity_124?
AssignVariableOp_124AssignVariableOp>assignvariableop_124_cond_1_adam_batch_normalization_53_beta_vIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_124q
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:2
Identity_125?
AssignVariableOp_125AssignVariableOp=assignvariableop_125_cond_1_adam_conv2d_transpose_11_kernel_vIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_125q
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:2
Identity_126?
AssignVariableOp_126AssignVariableOp;assignvariableop_126_cond_1_adam_conv2d_transpose_11_bias_vIdentity_126:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_126q
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:2
Identity_127?
AssignVariableOp_127AssignVariableOp3assignvariableop_127_cond_1_adam_conv2d_58_kernel_vIdentity_127:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_127q
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:2
Identity_128?
AssignVariableOp_128AssignVariableOp1assignvariableop_128_cond_1_adam_conv2d_58_bias_vIdentity_128:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_128q
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:2
Identity_129?
AssignVariableOp_129AssignVariableOp?assignvariableop_129_cond_1_adam_batch_normalization_55_gamma_vIdentity_129:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_129q
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:2
Identity_130?
AssignVariableOp_130AssignVariableOp>assignvariableop_130_cond_1_adam_batch_normalization_55_beta_vIdentity_130:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_130q
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:2
Identity_131?
AssignVariableOp_131AssignVariableOp3assignvariableop_131_cond_1_adam_conv2d_59_kernel_vIdentity_131:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_131q
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0*
_output_shapes
:2
Identity_132?
AssignVariableOp_132AssignVariableOp1assignvariableop_132_cond_1_adam_conv2d_59_bias_vIdentity_132:output:0"/device:CPU:0*
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
?
?
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_77424

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
?
?
6__inference_batch_normalization_53_layer_call_fn_80858

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
Q__inference_batch_normalization_53_layer_call_and_return_conditional_losses_770352
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
?
?
6__inference_batch_normalization_43_layer_call_fn_79979

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
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_772972
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
H__inference_activation_49_layer_call_and_return_conditional_losses_80542

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
?
c
E__inference_dropout_19_layer_call_and_return_conditional_losses_77509

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
3__inference_conv2d_transpose_10_layer_call_fn_76973

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
  ?E8? *W
fRRP
N__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_769632
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
?
K
/__inference_max_pooling2d_9_layer_call_fn_76441

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
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_764352
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
D__inference_conv2d_56_layer_call_and_return_conditional_losses_80798

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
?
?
6__inference_batch_normalization_43_layer_call_fn_79915

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
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_764182
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
?
d
H__inference_activation_45_layer_call_and_return_conditional_losses_80170

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
?
?
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_79953

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
?
?
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_80628

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
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_76503

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
?
d
E__inference_dropout_23_layer_call_and_return_conditional_losses_78144

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
Q__inference_batch_normalization_47_layer_call_and_return_conditional_losses_77569

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
d
E__inference_dropout_22_layer_call_and_return_conditional_losses_80771

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
?
F
*__inference_dropout_21_layer_call_fn_80587

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
  ?E8? *N
fIRG
E__inference_dropout_21_layer_call_and_return_conditional_losses_778192
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
?
?
6__inference_batch_normalization_55_layer_call_fn_81057

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
Q__inference_batch_normalization_55_layer_call_and_return_conditional_losses_782092
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
Q__inference_batch_normalization_47_layer_call_and_return_conditional_losses_76619

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
?
?
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_80075

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
3__inference_conv2d_transpose_11_layer_call_fn_77123

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
  ?E8? *W
fRRP
N__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_771132
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
?
d
E__inference_dropout_19_layer_call_and_return_conditional_losses_80187

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
?
?
6__inference_batch_normalization_51_layer_call_fn_80723

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
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_778792
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
?
c
E__inference_dropout_22_layer_call_and_return_conditional_losses_80776

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
?
?
D__inference_conv2d_58_layer_call_and_return_conditional_losses_78174

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
~
)__inference_conv2d_50_layer_call_fn_80223

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
D__inference_conv2d_50_layer_call_and_return_conditional_losses_775342
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
?
F
*__inference_dropout_23_layer_call_fn_80985

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
E__inference_dropout_23_layer_call_and_return_conditional_losses_781492
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
?
?
Q__inference_batch_normalization_53_layer_call_and_return_conditional_losses_80909

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
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_76418

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
E__inference_dropout_21_layer_call_and_return_conditional_losses_80572

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
?
~
)__inference_conv2d_54_layer_call_fn_80608

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
D__inference_conv2d_54_layer_call_and_return_conditional_losses_778442
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
?
?
6__inference_batch_normalization_53_layer_call_fn_80935

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
Q__inference_batch_normalization_53_layer_call_and_return_conditional_losses_780622
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
?
?
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_77442

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
?
c
*__inference_dropout_19_layer_call_fn_80197

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
  ?E8? *N
fIRG
E__inference_dropout_19_layer_call_and_return_conditional_losses_775042
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
?
?
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_80139

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
?
d
H__inference_activation_55_layer_call_and_return_conditional_losses_78268

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
Q__inference_batch_normalization_47_layer_call_and_return_conditional_losses_77587

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
?
?
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_79889

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
d
H__inference_activation_53_layer_call_and_return_conditional_losses_78103

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
?
I
-__inference_activation_47_layer_call_fn_80361

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
H__inference_activation_47_layer_call_and_return_conditional_losses_776282
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
?
F
*__inference_dropout_19_layer_call_fn_80202

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
  ?E8? *N
fIRG
E__inference_dropout_19_layer_call_and_return_conditional_losses_775092
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
?
?
D__inference_conv2d_46_layer_call_and_return_conditional_losses_79842

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
?
?
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_76735

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
?
d
H__inference_activation_53_layer_call_and_return_conditional_losses_80940

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
?
?
6__inference_batch_normalization_53_layer_call_fn_80922

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
Q__inference_batch_normalization_53_layer_call_and_return_conditional_losses_780442
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
c
*__inference_dropout_21_layer_call_fn_80582

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
  ?E8? *N
fIRG
E__inference_dropout_21_layer_call_and_return_conditional_losses_778142
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
?
~
)__inference_conv2d_59_layer_call_fn_81166

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
D__inference_conv2d_59_layer_call_and_return_conditional_losses_782892
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
?
?
Q__inference_batch_normalization_55_layer_call_and_return_conditional_losses_81090

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
?
?
Q__inference_batch_normalization_55_layer_call_and_return_conditional_losses_81108

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
D__inference_conv2d_50_layer_call_and_return_conditional_losses_77534

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
?
?
'__inference_model_3_layer_call_fn_78943
input_4
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
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
B__inference_model_3_layer_call_and_return_conditional_losses_788402
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
_user_specified_name	input_4
?
?
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_80493

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
?
c
E__inference_dropout_22_layer_call_and_return_conditional_losses_77984

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
f
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_76435

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
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_80511

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
?
?
D__inference_conv2d_46_layer_call_and_return_conditional_losses_77244

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
?
I
-__inference_activation_45_layer_call_fn_80175

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
H__inference_activation_45_layer_call_and_return_conditional_losses_774832
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
??
?
B__inference_model_3_layer_call_and_return_conditional_losses_78593

inputs
conv2d_46_78455
conv2d_46_78457 
batch_normalization_43_78460 
batch_normalization_43_78462 
batch_normalization_43_78464 
batch_normalization_43_78466
conv2d_48_78472
conv2d_48_78474 
batch_normalization_45_78477 
batch_normalization_45_78479 
batch_normalization_45_78481 
batch_normalization_45_78483
conv2d_50_78489
conv2d_50_78491 
batch_normalization_47_78494 
batch_normalization_47_78496 
batch_normalization_47_78498 
batch_normalization_47_78500
conv2d_52_78506
conv2d_52_78508 
batch_normalization_49_78511 
batch_normalization_49_78513 
batch_normalization_49_78515 
batch_normalization_49_78517
conv2d_transpose_9_78521
conv2d_transpose_9_78523
conv2d_54_78528
conv2d_54_78530 
batch_normalization_51_78533 
batch_normalization_51_78535 
batch_normalization_51_78537 
batch_normalization_51_78539
conv2d_transpose_10_78543
conv2d_transpose_10_78545
conv2d_56_78550
conv2d_56_78552 
batch_normalization_53_78555 
batch_normalization_53_78557 
batch_normalization_53_78559 
batch_normalization_53_78561
conv2d_transpose_11_78565
conv2d_transpose_11_78567
conv2d_58_78572
conv2d_58_78574 
batch_normalization_55_78577 
batch_normalization_55_78579 
batch_normalization_55_78581 
batch_normalization_55_78583
conv2d_59_78587
conv2d_59_78589
identity??.batch_normalization_43/StatefulPartitionedCall?.batch_normalization_45/StatefulPartitionedCall?.batch_normalization_47/StatefulPartitionedCall?.batch_normalization_49/StatefulPartitionedCall?.batch_normalization_51/StatefulPartitionedCall?.batch_normalization_53/StatefulPartitionedCall?.batch_normalization_55/StatefulPartitionedCall?!conv2d_46/StatefulPartitionedCall?!conv2d_48/StatefulPartitionedCall?!conv2d_50/StatefulPartitionedCall?!conv2d_52/StatefulPartitionedCall?!conv2d_54/StatefulPartitionedCall?!conv2d_56/StatefulPartitionedCall?!conv2d_58/StatefulPartitionedCall?!conv2d_59/StatefulPartitionedCall?+conv2d_transpose_10/StatefulPartitionedCall?+conv2d_transpose_11/StatefulPartitionedCall?*conv2d_transpose_9/StatefulPartitionedCall?"dropout_18/StatefulPartitionedCall?"dropout_19/StatefulPartitionedCall?"dropout_20/StatefulPartitionedCall?"dropout_21/StatefulPartitionedCall?"dropout_22/StatefulPartitionedCall?"dropout_23/StatefulPartitionedCallg
CastCastinputs*

DstT0*

SrcT0*1
_output_shapes
:???????????2
Cast?
!conv2d_46/StatefulPartitionedCallStatefulPartitionedCallCast:y:0conv2d_46_78455conv2d_46_78457*
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
D__inference_conv2d_46_layer_call_and_return_conditional_losses_772442#
!conv2d_46/StatefulPartitionedCall?
.batch_normalization_43/StatefulPartitionedCallStatefulPartitionedCall*conv2d_46/StatefulPartitionedCall:output:0batch_normalization_43_78460batch_normalization_43_78462batch_normalization_43_78464batch_normalization_43_78466*
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
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_7727920
.batch_normalization_43/StatefulPartitionedCall?
activation_43/PartitionedCallPartitionedCall7batch_normalization_43/StatefulPartitionedCall:output:0*
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
H__inference_activation_43_layer_call_and_return_conditional_losses_773382
activation_43/PartitionedCall?
max_pooling2d_9/PartitionedCallPartitionedCall&activation_43/PartitionedCall:output:0*
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
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_764352!
max_pooling2d_9/PartitionedCall?
"dropout_18/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_9/PartitionedCall:output:0*
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
  ?E8? *N
fIRG
E__inference_dropout_18_layer_call_and_return_conditional_losses_773592$
"dropout_18/StatefulPartitionedCall?
!conv2d_48/StatefulPartitionedCallStatefulPartitionedCall+dropout_18/StatefulPartitionedCall:output:0conv2d_48_78472conv2d_48_78474*
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
D__inference_conv2d_48_layer_call_and_return_conditional_losses_773892#
!conv2d_48/StatefulPartitionedCall?
.batch_normalization_45/StatefulPartitionedCallStatefulPartitionedCall*conv2d_48/StatefulPartitionedCall:output:0batch_normalization_45_78477batch_normalization_45_78479batch_normalization_45_78481batch_normalization_45_78483*
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
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_7742420
.batch_normalization_45/StatefulPartitionedCall?
activation_45/PartitionedCallPartitionedCall7batch_normalization_45/StatefulPartitionedCall:output:0*
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
H__inference_activation_45_layer_call_and_return_conditional_losses_774832
activation_45/PartitionedCall?
 max_pooling2d_10/PartitionedCallPartitionedCall&activation_45/PartitionedCall:output:0*
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
  ?E8? *T
fORM
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_765512"
 max_pooling2d_10/PartitionedCall?
"dropout_19/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_10/PartitionedCall:output:0#^dropout_18/StatefulPartitionedCall*
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
  ?E8? *N
fIRG
E__inference_dropout_19_layer_call_and_return_conditional_losses_775042$
"dropout_19/StatefulPartitionedCall?
!conv2d_50/StatefulPartitionedCallStatefulPartitionedCall+dropout_19/StatefulPartitionedCall:output:0conv2d_50_78489conv2d_50_78491*
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
D__inference_conv2d_50_layer_call_and_return_conditional_losses_775342#
!conv2d_50/StatefulPartitionedCall?
.batch_normalization_47/StatefulPartitionedCallStatefulPartitionedCall*conv2d_50/StatefulPartitionedCall:output:0batch_normalization_47_78494batch_normalization_47_78496batch_normalization_47_78498batch_normalization_47_78500*
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
Q__inference_batch_normalization_47_layer_call_and_return_conditional_losses_7756920
.batch_normalization_47/StatefulPartitionedCall?
activation_47/PartitionedCallPartitionedCall7batch_normalization_47/StatefulPartitionedCall:output:0*
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
H__inference_activation_47_layer_call_and_return_conditional_losses_776282
activation_47/PartitionedCall?
 max_pooling2d_11/PartitionedCallPartitionedCall&activation_47/PartitionedCall:output:0*
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
  ?E8? *T
fORM
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_766672"
 max_pooling2d_11/PartitionedCall?
"dropout_20/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_11/PartitionedCall:output:0#^dropout_19/StatefulPartitionedCall*
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
  ?E8? *N
fIRG
E__inference_dropout_20_layer_call_and_return_conditional_losses_776492$
"dropout_20/StatefulPartitionedCall?
!conv2d_52/StatefulPartitionedCallStatefulPartitionedCall+dropout_20/StatefulPartitionedCall:output:0conv2d_52_78506conv2d_52_78508*
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
D__inference_conv2d_52_layer_call_and_return_conditional_losses_776792#
!conv2d_52/StatefulPartitionedCall?
.batch_normalization_49/StatefulPartitionedCallStatefulPartitionedCall*conv2d_52/StatefulPartitionedCall:output:0batch_normalization_49_78511batch_normalization_49_78513batch_normalization_49_78515batch_normalization_49_78517*
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
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_7771420
.batch_normalization_49/StatefulPartitionedCall?
activation_49/PartitionedCallPartitionedCall7batch_normalization_49/StatefulPartitionedCall:output:0*
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
H__inference_activation_49_layer_call_and_return_conditional_losses_777732
activation_49/PartitionedCall?
*conv2d_transpose_9/StatefulPartitionedCallStatefulPartitionedCall&activation_49/PartitionedCall:output:0conv2d_transpose_9_78521conv2d_transpose_9_78523*
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
M__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_768132,
*conv2d_transpose_9/StatefulPartitionedCall?
concatenate_9/PartitionedCallPartitionedCall3conv2d_transpose_9/StatefulPartitionedCall:output:0&activation_47/PartitionedCall:output:0*
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
H__inference_concatenate_9_layer_call_and_return_conditional_losses_777932
concatenate_9/PartitionedCall?
"dropout_21/StatefulPartitionedCallStatefulPartitionedCall&concatenate_9/PartitionedCall:output:0#^dropout_20/StatefulPartitionedCall*
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
  ?E8? *N
fIRG
E__inference_dropout_21_layer_call_and_return_conditional_losses_778142$
"dropout_21/StatefulPartitionedCall?
!conv2d_54/StatefulPartitionedCallStatefulPartitionedCall+dropout_21/StatefulPartitionedCall:output:0conv2d_54_78528conv2d_54_78530*
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
D__inference_conv2d_54_layer_call_and_return_conditional_losses_778442#
!conv2d_54/StatefulPartitionedCall?
.batch_normalization_51/StatefulPartitionedCallStatefulPartitionedCall*conv2d_54/StatefulPartitionedCall:output:0batch_normalization_51_78533batch_normalization_51_78535batch_normalization_51_78537batch_normalization_51_78539*
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
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_7787920
.batch_normalization_51/StatefulPartitionedCall?
activation_51/PartitionedCallPartitionedCall7batch_normalization_51/StatefulPartitionedCall:output:0*
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
H__inference_activation_51_layer_call_and_return_conditional_losses_779382
activation_51/PartitionedCall?
+conv2d_transpose_10/StatefulPartitionedCallStatefulPartitionedCall&activation_51/PartitionedCall:output:0conv2d_transpose_10_78543conv2d_transpose_10_78545*
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
  ?E8? *W
fRRP
N__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_769632-
+conv2d_transpose_10/StatefulPartitionedCall?
concatenate_10/PartitionedCallPartitionedCall4conv2d_transpose_10/StatefulPartitionedCall:output:0&activation_45/PartitionedCall:output:0*
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
  ?E8? *R
fMRK
I__inference_concatenate_10_layer_call_and_return_conditional_losses_779582 
concatenate_10/PartitionedCall?
"dropout_22/StatefulPartitionedCallStatefulPartitionedCall'concatenate_10/PartitionedCall:output:0#^dropout_21/StatefulPartitionedCall*
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
E__inference_dropout_22_layer_call_and_return_conditional_losses_779792$
"dropout_22/StatefulPartitionedCall?
!conv2d_56/StatefulPartitionedCallStatefulPartitionedCall+dropout_22/StatefulPartitionedCall:output:0conv2d_56_78550conv2d_56_78552*
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
D__inference_conv2d_56_layer_call_and_return_conditional_losses_780092#
!conv2d_56/StatefulPartitionedCall?
.batch_normalization_53/StatefulPartitionedCallStatefulPartitionedCall*conv2d_56/StatefulPartitionedCall:output:0batch_normalization_53_78555batch_normalization_53_78557batch_normalization_53_78559batch_normalization_53_78561*
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
Q__inference_batch_normalization_53_layer_call_and_return_conditional_losses_7804420
.batch_normalization_53/StatefulPartitionedCall?
activation_53/PartitionedCallPartitionedCall7batch_normalization_53/StatefulPartitionedCall:output:0*
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
H__inference_activation_53_layer_call_and_return_conditional_losses_781032
activation_53/PartitionedCall?
+conv2d_transpose_11/StatefulPartitionedCallStatefulPartitionedCall&activation_53/PartitionedCall:output:0conv2d_transpose_11_78565conv2d_transpose_11_78567*
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
  ?E8? *W
fRRP
N__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_771132-
+conv2d_transpose_11/StatefulPartitionedCall?
concatenate_11/PartitionedCallPartitionedCall4conv2d_transpose_11/StatefulPartitionedCall:output:0&activation_43/PartitionedCall:output:0*
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
  ?E8? *R
fMRK
I__inference_concatenate_11_layer_call_and_return_conditional_losses_781232 
concatenate_11/PartitionedCall?
"dropout_23/StatefulPartitionedCallStatefulPartitionedCall'concatenate_11/PartitionedCall:output:0#^dropout_22/StatefulPartitionedCall*
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
E__inference_dropout_23_layer_call_and_return_conditional_losses_781442$
"dropout_23/StatefulPartitionedCall?
!conv2d_58/StatefulPartitionedCallStatefulPartitionedCall+dropout_23/StatefulPartitionedCall:output:0conv2d_58_78572conv2d_58_78574*
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
D__inference_conv2d_58_layer_call_and_return_conditional_losses_781742#
!conv2d_58/StatefulPartitionedCall?
.batch_normalization_55/StatefulPartitionedCallStatefulPartitionedCall*conv2d_58/StatefulPartitionedCall:output:0batch_normalization_55_78577batch_normalization_55_78579batch_normalization_55_78581batch_normalization_55_78583*
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
Q__inference_batch_normalization_55_layer_call_and_return_conditional_losses_7820920
.batch_normalization_55/StatefulPartitionedCall?
activation_55/PartitionedCallPartitionedCall7batch_normalization_55/StatefulPartitionedCall:output:0*
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
H__inference_activation_55_layer_call_and_return_conditional_losses_782682
activation_55/PartitionedCall?
!conv2d_59/StatefulPartitionedCallStatefulPartitionedCall&activation_55/PartitionedCall:output:0conv2d_59_78587conv2d_59_78589*
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
D__inference_conv2d_59_layer_call_and_return_conditional_losses_782892#
!conv2d_59/StatefulPartitionedCall?
IdentityIdentity*conv2d_59/StatefulPartitionedCall:output:0/^batch_normalization_43/StatefulPartitionedCall/^batch_normalization_45/StatefulPartitionedCall/^batch_normalization_47/StatefulPartitionedCall/^batch_normalization_49/StatefulPartitionedCall/^batch_normalization_51/StatefulPartitionedCall/^batch_normalization_53/StatefulPartitionedCall/^batch_normalization_55/StatefulPartitionedCall"^conv2d_46/StatefulPartitionedCall"^conv2d_48/StatefulPartitionedCall"^conv2d_50/StatefulPartitionedCall"^conv2d_52/StatefulPartitionedCall"^conv2d_54/StatefulPartitionedCall"^conv2d_56/StatefulPartitionedCall"^conv2d_58/StatefulPartitionedCall"^conv2d_59/StatefulPartitionedCall,^conv2d_transpose_10/StatefulPartitionedCall,^conv2d_transpose_11/StatefulPartitionedCall+^conv2d_transpose_9/StatefulPartitionedCall#^dropout_18/StatefulPartitionedCall#^dropout_19/StatefulPartitionedCall#^dropout_20/StatefulPartitionedCall#^dropout_21/StatefulPartitionedCall#^dropout_22/StatefulPartitionedCall#^dropout_23/StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_43/StatefulPartitionedCall.batch_normalization_43/StatefulPartitionedCall2`
.batch_normalization_45/StatefulPartitionedCall.batch_normalization_45/StatefulPartitionedCall2`
.batch_normalization_47/StatefulPartitionedCall.batch_normalization_47/StatefulPartitionedCall2`
.batch_normalization_49/StatefulPartitionedCall.batch_normalization_49/StatefulPartitionedCall2`
.batch_normalization_51/StatefulPartitionedCall.batch_normalization_51/StatefulPartitionedCall2`
.batch_normalization_53/StatefulPartitionedCall.batch_normalization_53/StatefulPartitionedCall2`
.batch_normalization_55/StatefulPartitionedCall.batch_normalization_55/StatefulPartitionedCall2F
!conv2d_46/StatefulPartitionedCall!conv2d_46/StatefulPartitionedCall2F
!conv2d_48/StatefulPartitionedCall!conv2d_48/StatefulPartitionedCall2F
!conv2d_50/StatefulPartitionedCall!conv2d_50/StatefulPartitionedCall2F
!conv2d_52/StatefulPartitionedCall!conv2d_52/StatefulPartitionedCall2F
!conv2d_54/StatefulPartitionedCall!conv2d_54/StatefulPartitionedCall2F
!conv2d_56/StatefulPartitionedCall!conv2d_56/StatefulPartitionedCall2F
!conv2d_58/StatefulPartitionedCall!conv2d_58/StatefulPartitionedCall2F
!conv2d_59/StatefulPartitionedCall!conv2d_59/StatefulPartitionedCall2Z
+conv2d_transpose_10/StatefulPartitionedCall+conv2d_transpose_10/StatefulPartitionedCall2Z
+conv2d_transpose_11/StatefulPartitionedCall+conv2d_transpose_11/StatefulPartitionedCall2X
*conv2d_transpose_9/StatefulPartitionedCall*conv2d_transpose_9/StatefulPartitionedCall2H
"dropout_18/StatefulPartitionedCall"dropout_18/StatefulPartitionedCall2H
"dropout_19/StatefulPartitionedCall"dropout_19/StatefulPartitionedCall2H
"dropout_20/StatefulPartitionedCall"dropout_20/StatefulPartitionedCall2H
"dropout_21/StatefulPartitionedCall"dropout_21/StatefulPartitionedCall2H
"dropout_22/StatefulPartitionedCall"dropout_22/StatefulPartitionedCall2H
"dropout_23/StatefulPartitionedCall"dropout_23/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_49_layer_call_fn_80524

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
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_767352
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
?
?
6__inference_batch_normalization_55_layer_call_fn_81070

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
Q__inference_batch_normalization_55_layer_call_and_return_conditional_losses_782272
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
??
?,
B__inference_model_3_layer_call_and_return_conditional_losses_79367

inputs,
(conv2d_46_conv2d_readvariableop_resource-
)conv2d_46_biasadd_readvariableop_resource2
.batch_normalization_43_readvariableop_resource4
0batch_normalization_43_readvariableop_1_resourceC
?batch_normalization_43_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_43_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_48_conv2d_readvariableop_resource-
)conv2d_48_biasadd_readvariableop_resource2
.batch_normalization_45_readvariableop_resource4
0batch_normalization_45_readvariableop_1_resourceC
?batch_normalization_45_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_45_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_50_conv2d_readvariableop_resource-
)conv2d_50_biasadd_readvariableop_resource2
.batch_normalization_47_readvariableop_resource4
0batch_normalization_47_readvariableop_1_resourceC
?batch_normalization_47_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_47_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_52_conv2d_readvariableop_resource-
)conv2d_52_biasadd_readvariableop_resource2
.batch_normalization_49_readvariableop_resource4
0batch_normalization_49_readvariableop_1_resourceC
?batch_normalization_49_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_49_fusedbatchnormv3_readvariableop_1_resource?
;conv2d_transpose_9_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_9_biasadd_readvariableop_resource,
(conv2d_54_conv2d_readvariableop_resource-
)conv2d_54_biasadd_readvariableop_resource2
.batch_normalization_51_readvariableop_resource4
0batch_normalization_51_readvariableop_1_resourceC
?batch_normalization_51_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_51_fusedbatchnormv3_readvariableop_1_resource@
<conv2d_transpose_10_conv2d_transpose_readvariableop_resource7
3conv2d_transpose_10_biasadd_readvariableop_resource,
(conv2d_56_conv2d_readvariableop_resource-
)conv2d_56_biasadd_readvariableop_resource2
.batch_normalization_53_readvariableop_resource4
0batch_normalization_53_readvariableop_1_resourceC
?batch_normalization_53_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_53_fusedbatchnormv3_readvariableop_1_resource@
<conv2d_transpose_11_conv2d_transpose_readvariableop_resource7
3conv2d_transpose_11_biasadd_readvariableop_resource,
(conv2d_58_conv2d_readvariableop_resource-
)conv2d_58_biasadd_readvariableop_resource2
.batch_normalization_55_readvariableop_resource4
0batch_normalization_55_readvariableop_1_resourceC
?batch_normalization_55_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_55_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_59_conv2d_readvariableop_resource-
)conv2d_59_biasadd_readvariableop_resource
identity??%batch_normalization_43/AssignNewValue?'batch_normalization_43/AssignNewValue_1?6batch_normalization_43/FusedBatchNormV3/ReadVariableOp?8batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_43/ReadVariableOp?'batch_normalization_43/ReadVariableOp_1?%batch_normalization_45/AssignNewValue?'batch_normalization_45/AssignNewValue_1?6batch_normalization_45/FusedBatchNormV3/ReadVariableOp?8batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_45/ReadVariableOp?'batch_normalization_45/ReadVariableOp_1?%batch_normalization_47/AssignNewValue?'batch_normalization_47/AssignNewValue_1?6batch_normalization_47/FusedBatchNormV3/ReadVariableOp?8batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_47/ReadVariableOp?'batch_normalization_47/ReadVariableOp_1?%batch_normalization_49/AssignNewValue?'batch_normalization_49/AssignNewValue_1?6batch_normalization_49/FusedBatchNormV3/ReadVariableOp?8batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_49/ReadVariableOp?'batch_normalization_49/ReadVariableOp_1?%batch_normalization_51/AssignNewValue?'batch_normalization_51/AssignNewValue_1?6batch_normalization_51/FusedBatchNormV3/ReadVariableOp?8batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_51/ReadVariableOp?'batch_normalization_51/ReadVariableOp_1?%batch_normalization_53/AssignNewValue?'batch_normalization_53/AssignNewValue_1?6batch_normalization_53/FusedBatchNormV3/ReadVariableOp?8batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_53/ReadVariableOp?'batch_normalization_53/ReadVariableOp_1?%batch_normalization_55/AssignNewValue?'batch_normalization_55/AssignNewValue_1?6batch_normalization_55/FusedBatchNormV3/ReadVariableOp?8batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_55/ReadVariableOp?'batch_normalization_55/ReadVariableOp_1? conv2d_46/BiasAdd/ReadVariableOp?conv2d_46/Conv2D/ReadVariableOp? conv2d_48/BiasAdd/ReadVariableOp?conv2d_48/Conv2D/ReadVariableOp? conv2d_50/BiasAdd/ReadVariableOp?conv2d_50/Conv2D/ReadVariableOp? conv2d_52/BiasAdd/ReadVariableOp?conv2d_52/Conv2D/ReadVariableOp? conv2d_54/BiasAdd/ReadVariableOp?conv2d_54/Conv2D/ReadVariableOp? conv2d_56/BiasAdd/ReadVariableOp?conv2d_56/Conv2D/ReadVariableOp? conv2d_58/BiasAdd/ReadVariableOp?conv2d_58/Conv2D/ReadVariableOp? conv2d_59/BiasAdd/ReadVariableOp?conv2d_59/Conv2D/ReadVariableOp?*conv2d_transpose_10/BiasAdd/ReadVariableOp?3conv2d_transpose_10/conv2d_transpose/ReadVariableOp?*conv2d_transpose_11/BiasAdd/ReadVariableOp?3conv2d_transpose_11/conv2d_transpose/ReadVariableOp?)conv2d_transpose_9/BiasAdd/ReadVariableOp?2conv2d_transpose_9/conv2d_transpose/ReadVariableOpg
CastCastinputs*

DstT0*

SrcT0*1
_output_shapes
:???????????2
Cast?
conv2d_46/Conv2D/ReadVariableOpReadVariableOp(conv2d_46_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_46/Conv2D/ReadVariableOp?
conv2d_46/Conv2D/CastCast'conv2d_46/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: 2
conv2d_46/Conv2D/Cast?
conv2d_46/Conv2DConv2DCast:y:0conv2d_46/Conv2D/Cast:y:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
conv2d_46/Conv2D?
 conv2d_46/BiasAdd/ReadVariableOpReadVariableOp)conv2d_46_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_46/BiasAdd/ReadVariableOp?
conv2d_46/BiasAdd/CastCast(conv2d_46/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: 2
conv2d_46/BiasAdd/Cast?
conv2d_46/BiasAddBiasAddconv2d_46/Conv2D:output:0conv2d_46/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:??????????? 2
conv2d_46/BiasAdd?
%batch_normalization_43/ReadVariableOpReadVariableOp.batch_normalization_43_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_43/ReadVariableOp?
'batch_normalization_43/ReadVariableOp_1ReadVariableOp0batch_normalization_43_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_43/ReadVariableOp_1?
6batch_normalization_43/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_43_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_43/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_43_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_43/FusedBatchNormV3FusedBatchNormV3conv2d_46/BiasAdd:output:0-batch_normalization_43/ReadVariableOp:value:0/batch_normalization_43/ReadVariableOp_1:value:0>batch_normalization_43/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_43/FusedBatchNormV3?
%batch_normalization_43/AssignNewValueAssignVariableOp?batch_normalization_43_fusedbatchnormv3_readvariableop_resource4batch_normalization_43/FusedBatchNormV3:batch_mean:07^batch_normalization_43/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*R
_classH
FDloc:@batch_normalization_43/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_43/AssignNewValue?
'batch_normalization_43/AssignNewValue_1AssignVariableOpAbatch_normalization_43_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_43/FusedBatchNormV3:batch_variance:09^batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*T
_classJ
HFloc:@batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_43/AssignNewValue_1?
activation_43/ReluRelu+batch_normalization_43/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? 2
activation_43/Relu?
max_pooling2d_9/MaxPoolMaxPool activation_43/Relu:activations:0*
T0*1
_output_shapes
:??????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_9/MaxPoolw
dropout_18/dropout/ConstConst*
_output_shapes
: *
dtype0*
value
B j?x2
dropout_18/dropout/Const?
dropout_18/dropout/MulMul max_pooling2d_9/MaxPool:output:0!dropout_18/dropout/Const:output:0*
T0*1
_output_shapes
:??????????? 2
dropout_18/dropout/Mul?
dropout_18/dropout/ShapeShape max_pooling2d_9/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_18/dropout/Shape?
/dropout_18/dropout/random_uniform/RandomUniformRandomUniform!dropout_18/dropout/Shape:output:0*
T0*1
_output_shapes
:??????????? *
dtype021
/dropout_18/dropout/random_uniform/RandomUniform?
!dropout_18/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value
B j?\2#
!dropout_18/dropout/GreaterEqual/y?
dropout_18/dropout/GreaterEqualGreaterEqual8dropout_18/dropout/random_uniform/RandomUniform:output:0*dropout_18/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:??????????? 2!
dropout_18/dropout/GreaterEqual?
dropout_18/dropout/CastCast#dropout_18/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:??????????? 2
dropout_18/dropout/Cast?
dropout_18/dropout/Mul_1Muldropout_18/dropout/Mul:z:0dropout_18/dropout/Cast:y:0*
T0*1
_output_shapes
:??????????? 2
dropout_18/dropout/Mul_1?
conv2d_48/Conv2D/ReadVariableOpReadVariableOp(conv2d_48_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_48/Conv2D/ReadVariableOp?
conv2d_48/Conv2D/CastCast'conv2d_48/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: @2
conv2d_48/Conv2D/Cast?
conv2d_48/Conv2DConv2Ddropout_18/dropout/Mul_1:z:0conv2d_48/Conv2D/Cast:y:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv2d_48/Conv2D?
 conv2d_48/BiasAdd/ReadVariableOpReadVariableOp)conv2d_48_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_48/BiasAdd/ReadVariableOp?
conv2d_48/BiasAdd/CastCast(conv2d_48/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@2
conv2d_48/BiasAdd/Cast?
conv2d_48/BiasAddBiasAddconv2d_48/Conv2D:output:0conv2d_48/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:???????????@2
conv2d_48/BiasAdd?
%batch_normalization_45/ReadVariableOpReadVariableOp.batch_normalization_45_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_45/ReadVariableOp?
'batch_normalization_45/ReadVariableOp_1ReadVariableOp0batch_normalization_45_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_45/ReadVariableOp_1?
6batch_normalization_45/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_45_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_45/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_45_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_45/FusedBatchNormV3FusedBatchNormV3conv2d_48/BiasAdd:output:0-batch_normalization_45/ReadVariableOp:value:0/batch_normalization_45/ReadVariableOp_1:value:0>batch_normalization_45/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_45/FusedBatchNormV3?
%batch_normalization_45/AssignNewValueAssignVariableOp?batch_normalization_45_fusedbatchnormv3_readvariableop_resource4batch_normalization_45/FusedBatchNormV3:batch_mean:07^batch_normalization_45/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*R
_classH
FDloc:@batch_normalization_45/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_45/AssignNewValue?
'batch_normalization_45/AssignNewValue_1AssignVariableOpAbatch_normalization_45_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_45/FusedBatchNormV3:batch_variance:09^batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*T
_classJ
HFloc:@batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_45/AssignNewValue_1?
activation_45/ReluRelu+batch_normalization_45/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????@2
activation_45/Relu?
max_pooling2d_10/MaxPoolMaxPool activation_45/Relu:activations:0*
T0*1
_output_shapes
:???????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_10/MaxPoolw
dropout_19/dropout/ConstConst*
_output_shapes
: *
dtype0*
value
B j?x2
dropout_19/dropout/Const?
dropout_19/dropout/MulMul!max_pooling2d_10/MaxPool:output:0!dropout_19/dropout/Const:output:0*
T0*1
_output_shapes
:???????????@2
dropout_19/dropout/Mul?
dropout_19/dropout/ShapeShape!max_pooling2d_10/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_19/dropout/Shape?
/dropout_19/dropout/random_uniform/RandomUniformRandomUniform!dropout_19/dropout/Shape:output:0*
T0*1
_output_shapes
:???????????@*
dtype021
/dropout_19/dropout/random_uniform/RandomUniform?
!dropout_19/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value
B j?\2#
!dropout_19/dropout/GreaterEqual/y?
dropout_19/dropout/GreaterEqualGreaterEqual8dropout_19/dropout/random_uniform/RandomUniform:output:0*dropout_19/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:???????????@2!
dropout_19/dropout/GreaterEqual?
dropout_19/dropout/CastCast#dropout_19/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:???????????@2
dropout_19/dropout/Cast?
dropout_19/dropout/Mul_1Muldropout_19/dropout/Mul:z:0dropout_19/dropout/Cast:y:0*
T0*1
_output_shapes
:???????????@2
dropout_19/dropout/Mul_1?
conv2d_50/Conv2D/ReadVariableOpReadVariableOp(conv2d_50_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02!
conv2d_50/Conv2D/ReadVariableOp?
conv2d_50/Conv2D/CastCast'conv2d_50/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*'
_output_shapes
:@?2
conv2d_50/Conv2D/Cast?
conv2d_50/Conv2DConv2Ddropout_19/dropout/Mul_1:z:0conv2d_50/Conv2D/Cast:y:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
conv2d_50/Conv2D?
 conv2d_50/BiasAdd/ReadVariableOpReadVariableOp)conv2d_50_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_50/BiasAdd/ReadVariableOp?
conv2d_50/BiasAdd/CastCast(conv2d_50/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:?2
conv2d_50/BiasAdd/Cast?
conv2d_50/BiasAddBiasAddconv2d_50/Conv2D:output:0conv2d_50/BiasAdd/Cast:y:0*
T0*2
_output_shapes 
:????????????2
conv2d_50/BiasAdd?
%batch_normalization_47/ReadVariableOpReadVariableOp.batch_normalization_47_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_47/ReadVariableOp?
'batch_normalization_47/ReadVariableOp_1ReadVariableOp0batch_normalization_47_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_47/ReadVariableOp_1?
6batch_normalization_47/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_47_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_47/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_47_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_47/FusedBatchNormV3FusedBatchNormV3conv2d_50/BiasAdd:output:0-batch_normalization_47/ReadVariableOp:value:0/batch_normalization_47/ReadVariableOp_1:value:0>batch_normalization_47/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_47/FusedBatchNormV3?
%batch_normalization_47/AssignNewValueAssignVariableOp?batch_normalization_47_fusedbatchnormv3_readvariableop_resource4batch_normalization_47/FusedBatchNormV3:batch_mean:07^batch_normalization_47/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*R
_classH
FDloc:@batch_normalization_47/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_47/AssignNewValue?
'batch_normalization_47/AssignNewValue_1AssignVariableOpAbatch_normalization_47_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_47/FusedBatchNormV3:batch_variance:09^batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*T
_classJ
HFloc:@batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_47/AssignNewValue_1?
activation_47/ReluRelu+batch_normalization_47/FusedBatchNormV3:y:0*
T0*2
_output_shapes 
:????????????2
activation_47/Relu?
max_pooling2d_11/MaxPoolMaxPool activation_47/Relu:activations:0*
T0*0
_output_shapes
:?????????D\?*
ksize
*
paddingVALID*
strides
2
max_pooling2d_11/MaxPoolw
dropout_20/dropout/ConstConst*
_output_shapes
: *
dtype0*
value
B j?x2
dropout_20/dropout/Const?
dropout_20/dropout/MulMul!max_pooling2d_11/MaxPool:output:0!dropout_20/dropout/Const:output:0*
T0*0
_output_shapes
:?????????D\?2
dropout_20/dropout/Mul?
dropout_20/dropout/ShapeShape!max_pooling2d_11/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_20/dropout/Shape?
/dropout_20/dropout/random_uniform/RandomUniformRandomUniform!dropout_20/dropout/Shape:output:0*
T0*0
_output_shapes
:?????????D\?*
dtype021
/dropout_20/dropout/random_uniform/RandomUniform?
!dropout_20/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value
B j?\2#
!dropout_20/dropout/GreaterEqual/y?
dropout_20/dropout/GreaterEqualGreaterEqual8dropout_20/dropout/random_uniform/RandomUniform:output:0*dropout_20/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:?????????D\?2!
dropout_20/dropout/GreaterEqual?
dropout_20/dropout/CastCast#dropout_20/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:?????????D\?2
dropout_20/dropout/Cast?
dropout_20/dropout/Mul_1Muldropout_20/dropout/Mul:z:0dropout_20/dropout/Cast:y:0*
T0*0
_output_shapes
:?????????D\?2
dropout_20/dropout/Mul_1?
conv2d_52/Conv2D/ReadVariableOpReadVariableOp(conv2d_52_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_52/Conv2D/ReadVariableOp?
conv2d_52/Conv2D/CastCast'conv2d_52/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*(
_output_shapes
:??2
conv2d_52/Conv2D/Cast?
conv2d_52/Conv2DConv2Ddropout_20/dropout/Mul_1:z:0conv2d_52/Conv2D/Cast:y:0*
T0*0
_output_shapes
:?????????D\?*
paddingSAME*
strides
2
conv2d_52/Conv2D?
 conv2d_52/BiasAdd/ReadVariableOpReadVariableOp)conv2d_52_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_52/BiasAdd/ReadVariableOp?
conv2d_52/BiasAdd/CastCast(conv2d_52/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:?2
conv2d_52/BiasAdd/Cast?
conv2d_52/BiasAddBiasAddconv2d_52/Conv2D:output:0conv2d_52/BiasAdd/Cast:y:0*
T0*0
_output_shapes
:?????????D\?2
conv2d_52/BiasAdd?
%batch_normalization_49/ReadVariableOpReadVariableOp.batch_normalization_49_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_49/ReadVariableOp?
'batch_normalization_49/ReadVariableOp_1ReadVariableOp0batch_normalization_49_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_49/ReadVariableOp_1?
6batch_normalization_49/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_49_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_49/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_49_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_49/FusedBatchNormV3FusedBatchNormV3conv2d_52/BiasAdd:output:0-batch_normalization_49/ReadVariableOp:value:0/batch_normalization_49/ReadVariableOp_1:value:0>batch_normalization_49/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????D\?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_49/FusedBatchNormV3?
%batch_normalization_49/AssignNewValueAssignVariableOp?batch_normalization_49_fusedbatchnormv3_readvariableop_resource4batch_normalization_49/FusedBatchNormV3:batch_mean:07^batch_normalization_49/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*R
_classH
FDloc:@batch_normalization_49/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_49/AssignNewValue?
'batch_normalization_49/AssignNewValue_1AssignVariableOpAbatch_normalization_49_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_49/FusedBatchNormV3:batch_variance:09^batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*T
_classJ
HFloc:@batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_49/AssignNewValue_1?
activation_49/ReluRelu+batch_normalization_49/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????D\?2
activation_49/Relu?
conv2d_transpose_9/ShapeShape activation_49/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_9/Shape?
&conv2d_transpose_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_9/strided_slice/stack?
(conv2d_transpose_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_9/strided_slice/stack_1?
(conv2d_transpose_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_9/strided_slice/stack_2?
 conv2d_transpose_9/strided_sliceStridedSlice!conv2d_transpose_9/Shape:output:0/conv2d_transpose_9/strided_slice/stack:output:01conv2d_transpose_9/strided_slice/stack_1:output:01conv2d_transpose_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_9/strided_slice{
conv2d_transpose_9/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_9/stack/1{
conv2d_transpose_9/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_9/stack/2{
conv2d_transpose_9/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_9/stack/3?
conv2d_transpose_9/stackPack)conv2d_transpose_9/strided_slice:output:0#conv2d_transpose_9/stack/1:output:0#conv2d_transpose_9/stack/2:output:0#conv2d_transpose_9/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_9/stack?
(conv2d_transpose_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_9/strided_slice_1/stack?
*conv2d_transpose_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_9/strided_slice_1/stack_1?
*conv2d_transpose_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_9/strided_slice_1/stack_2?
"conv2d_transpose_9/strided_slice_1StridedSlice!conv2d_transpose_9/stack:output:01conv2d_transpose_9/strided_slice_1/stack:output:03conv2d_transpose_9/strided_slice_1/stack_1:output:03conv2d_transpose_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_9/strided_slice_1?
2conv2d_transpose_9/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_9_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype024
2conv2d_transpose_9/conv2d_transpose/ReadVariableOp?
(conv2d_transpose_9/conv2d_transpose/CastCast:conv2d_transpose_9/conv2d_transpose/ReadVariableOp:value:0*

DstT0*

SrcT0*(
_output_shapes
:??2*
(conv2d_transpose_9/conv2d_transpose/Cast?
#conv2d_transpose_9/conv2d_transposeConv2DBackpropInput!conv2d_transpose_9/stack:output:0,conv2d_transpose_9/conv2d_transpose/Cast:y:0 activation_49/Relu:activations:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2%
#conv2d_transpose_9/conv2d_transpose?
)conv2d_transpose_9/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_9_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)conv2d_transpose_9/BiasAdd/ReadVariableOp?
conv2d_transpose_9/BiasAdd/CastCast1conv2d_transpose_9/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:?2!
conv2d_transpose_9/BiasAdd/Cast?
conv2d_transpose_9/BiasAddBiasAdd,conv2d_transpose_9/conv2d_transpose:output:0#conv2d_transpose_9/BiasAdd/Cast:y:0*
T0*2
_output_shapes 
:????????????2
conv2d_transpose_9/BiasAddx
concatenate_9/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_9/concat/axis?
concatenate_9/concatConcatV2#conv2d_transpose_9/BiasAdd:output:0 activation_47/Relu:activations:0"concatenate_9/concat/axis:output:0*
N*
T0*2
_output_shapes 
:????????????2
concatenate_9/concatw
dropout_21/dropout/ConstConst*
_output_shapes
: *
dtype0*
value
B j?x2
dropout_21/dropout/Const?
dropout_21/dropout/MulMulconcatenate_9/concat:output:0!dropout_21/dropout/Const:output:0*
T0*2
_output_shapes 
:????????????2
dropout_21/dropout/Mul?
dropout_21/dropout/ShapeShapeconcatenate_9/concat:output:0*
T0*
_output_shapes
:2
dropout_21/dropout/Shape?
/dropout_21/dropout/random_uniform/RandomUniformRandomUniform!dropout_21/dropout/Shape:output:0*
T0*2
_output_shapes 
:????????????*
dtype021
/dropout_21/dropout/random_uniform/RandomUniform?
!dropout_21/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value
B j?\2#
!dropout_21/dropout/GreaterEqual/y?
dropout_21/dropout/GreaterEqualGreaterEqual8dropout_21/dropout/random_uniform/RandomUniform:output:0*dropout_21/dropout/GreaterEqual/y:output:0*
T0*2
_output_shapes 
:????????????2!
dropout_21/dropout/GreaterEqual?
dropout_21/dropout/CastCast#dropout_21/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*2
_output_shapes 
:????????????2
dropout_21/dropout/Cast?
dropout_21/dropout/Mul_1Muldropout_21/dropout/Mul:z:0dropout_21/dropout/Cast:y:0*
T0*2
_output_shapes 
:????????????2
dropout_21/dropout/Mul_1?
conv2d_54/Conv2D/ReadVariableOpReadVariableOp(conv2d_54_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_54/Conv2D/ReadVariableOp?
conv2d_54/Conv2D/CastCast'conv2d_54/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*(
_output_shapes
:??2
conv2d_54/Conv2D/Cast?
conv2d_54/Conv2DConv2Ddropout_21/dropout/Mul_1:z:0conv2d_54/Conv2D/Cast:y:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
conv2d_54/Conv2D?
 conv2d_54/BiasAdd/ReadVariableOpReadVariableOp)conv2d_54_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_54/BiasAdd/ReadVariableOp?
conv2d_54/BiasAdd/CastCast(conv2d_54/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:?2
conv2d_54/BiasAdd/Cast?
conv2d_54/BiasAddBiasAddconv2d_54/Conv2D:output:0conv2d_54/BiasAdd/Cast:y:0*
T0*2
_output_shapes 
:????????????2
conv2d_54/BiasAdd?
%batch_normalization_51/ReadVariableOpReadVariableOp.batch_normalization_51_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_51/ReadVariableOp?
'batch_normalization_51/ReadVariableOp_1ReadVariableOp0batch_normalization_51_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_51/ReadVariableOp_1?
6batch_normalization_51/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_51_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_51/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_51_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_51/FusedBatchNormV3FusedBatchNormV3conv2d_54/BiasAdd:output:0-batch_normalization_51/ReadVariableOp:value:0/batch_normalization_51/ReadVariableOp_1:value:0>batch_normalization_51/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_51/FusedBatchNormV3?
%batch_normalization_51/AssignNewValueAssignVariableOp?batch_normalization_51_fusedbatchnormv3_readvariableop_resource4batch_normalization_51/FusedBatchNormV3:batch_mean:07^batch_normalization_51/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*R
_classH
FDloc:@batch_normalization_51/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_51/AssignNewValue?
'batch_normalization_51/AssignNewValue_1AssignVariableOpAbatch_normalization_51_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_51/FusedBatchNormV3:batch_variance:09^batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*T
_classJ
HFloc:@batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_51/AssignNewValue_1?
activation_51/ReluRelu+batch_normalization_51/FusedBatchNormV3:y:0*
T0*2
_output_shapes 
:????????????2
activation_51/Relu?
conv2d_transpose_10/ShapeShape activation_51/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_10/Shape?
'conv2d_transpose_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_10/strided_slice/stack?
)conv2d_transpose_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_10/strided_slice/stack_1?
)conv2d_transpose_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_10/strided_slice/stack_2?
!conv2d_transpose_10/strided_sliceStridedSlice"conv2d_transpose_10/Shape:output:00conv2d_transpose_10/strided_slice/stack:output:02conv2d_transpose_10/strided_slice/stack_1:output:02conv2d_transpose_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_10/strided_slice}
conv2d_transpose_10/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_10/stack/1}
conv2d_transpose_10/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_10/stack/2|
conv2d_transpose_10/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_10/stack/3?
conv2d_transpose_10/stackPack*conv2d_transpose_10/strided_slice:output:0$conv2d_transpose_10/stack/1:output:0$conv2d_transpose_10/stack/2:output:0$conv2d_transpose_10/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_10/stack?
)conv2d_transpose_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_10/strided_slice_1/stack?
+conv2d_transpose_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_10/strided_slice_1/stack_1?
+conv2d_transpose_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_10/strided_slice_1/stack_2?
#conv2d_transpose_10/strided_slice_1StridedSlice"conv2d_transpose_10/stack:output:02conv2d_transpose_10/strided_slice_1/stack:output:04conv2d_transpose_10/strided_slice_1/stack_1:output:04conv2d_transpose_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_10/strided_slice_1?
3conv2d_transpose_10/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_10_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype025
3conv2d_transpose_10/conv2d_transpose/ReadVariableOp?
)conv2d_transpose_10/conv2d_transpose/CastCast;conv2d_transpose_10/conv2d_transpose/ReadVariableOp:value:0*

DstT0*

SrcT0*'
_output_shapes
:@?2+
)conv2d_transpose_10/conv2d_transpose/Cast?
$conv2d_transpose_10/conv2d_transposeConv2DBackpropInput"conv2d_transpose_10/stack:output:0-conv2d_transpose_10/conv2d_transpose/Cast:y:0 activation_51/Relu:activations:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2&
$conv2d_transpose_10/conv2d_transpose?
*conv2d_transpose_10/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*conv2d_transpose_10/BiasAdd/ReadVariableOp?
 conv2d_transpose_10/BiasAdd/CastCast2conv2d_transpose_10/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@2"
 conv2d_transpose_10/BiasAdd/Cast?
conv2d_transpose_10/BiasAddBiasAdd-conv2d_transpose_10/conv2d_transpose:output:0$conv2d_transpose_10/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:???????????@2
conv2d_transpose_10/BiasAddz
concatenate_10/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_10/concat/axis?
concatenate_10/concatConcatV2$conv2d_transpose_10/BiasAdd:output:0 activation_45/Relu:activations:0#concatenate_10/concat/axis:output:0*
N*
T0*2
_output_shapes 
:????????????2
concatenate_10/concatw
dropout_22/dropout/ConstConst*
_output_shapes
: *
dtype0*
value
B j?x2
dropout_22/dropout/Const?
dropout_22/dropout/MulMulconcatenate_10/concat:output:0!dropout_22/dropout/Const:output:0*
T0*2
_output_shapes 
:????????????2
dropout_22/dropout/Mul?
dropout_22/dropout/ShapeShapeconcatenate_10/concat:output:0*
T0*
_output_shapes
:2
dropout_22/dropout/Shape?
/dropout_22/dropout/random_uniform/RandomUniformRandomUniform!dropout_22/dropout/Shape:output:0*
T0*2
_output_shapes 
:????????????*
dtype021
/dropout_22/dropout/random_uniform/RandomUniform?
!dropout_22/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value
B j?\2#
!dropout_22/dropout/GreaterEqual/y?
dropout_22/dropout/GreaterEqualGreaterEqual8dropout_22/dropout/random_uniform/RandomUniform:output:0*dropout_22/dropout/GreaterEqual/y:output:0*
T0*2
_output_shapes 
:????????????2!
dropout_22/dropout/GreaterEqual?
dropout_22/dropout/CastCast#dropout_22/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*2
_output_shapes 
:????????????2
dropout_22/dropout/Cast?
dropout_22/dropout/Mul_1Muldropout_22/dropout/Mul:z:0dropout_22/dropout/Cast:y:0*
T0*2
_output_shapes 
:????????????2
dropout_22/dropout/Mul_1?
conv2d_56/Conv2D/ReadVariableOpReadVariableOp(conv2d_56_conv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02!
conv2d_56/Conv2D/ReadVariableOp?
conv2d_56/Conv2D/CastCast'conv2d_56/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*'
_output_shapes
:?@2
conv2d_56/Conv2D/Cast?
conv2d_56/Conv2DConv2Ddropout_22/dropout/Mul_1:z:0conv2d_56/Conv2D/Cast:y:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv2d_56/Conv2D?
 conv2d_56/BiasAdd/ReadVariableOpReadVariableOp)conv2d_56_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_56/BiasAdd/ReadVariableOp?
conv2d_56/BiasAdd/CastCast(conv2d_56/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@2
conv2d_56/BiasAdd/Cast?
conv2d_56/BiasAddBiasAddconv2d_56/Conv2D:output:0conv2d_56/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:???????????@2
conv2d_56/BiasAdd?
%batch_normalization_53/ReadVariableOpReadVariableOp.batch_normalization_53_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_53/ReadVariableOp?
'batch_normalization_53/ReadVariableOp_1ReadVariableOp0batch_normalization_53_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_53/ReadVariableOp_1?
6batch_normalization_53/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_53_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_53/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_53_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_53/FusedBatchNormV3FusedBatchNormV3conv2d_56/BiasAdd:output:0-batch_normalization_53/ReadVariableOp:value:0/batch_normalization_53/ReadVariableOp_1:value:0>batch_normalization_53/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_53/FusedBatchNormV3?
%batch_normalization_53/AssignNewValueAssignVariableOp?batch_normalization_53_fusedbatchnormv3_readvariableop_resource4batch_normalization_53/FusedBatchNormV3:batch_mean:07^batch_normalization_53/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*R
_classH
FDloc:@batch_normalization_53/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_53/AssignNewValue?
'batch_normalization_53/AssignNewValue_1AssignVariableOpAbatch_normalization_53_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_53/FusedBatchNormV3:batch_variance:09^batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*T
_classJ
HFloc:@batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_53/AssignNewValue_1?
activation_53/ReluRelu+batch_normalization_53/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????@2
activation_53/Relu?
conv2d_transpose_11/ShapeShape activation_53/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_11/Shape?
'conv2d_transpose_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_11/strided_slice/stack?
)conv2d_transpose_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_11/strided_slice/stack_1?
)conv2d_transpose_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_11/strided_slice/stack_2?
!conv2d_transpose_11/strided_sliceStridedSlice"conv2d_transpose_11/Shape:output:00conv2d_transpose_11/strided_slice/stack:output:02conv2d_transpose_11/strided_slice/stack_1:output:02conv2d_transpose_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_11/strided_slice}
conv2d_transpose_11/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_11/stack/1}
conv2d_transpose_11/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_11/stack/2|
conv2d_transpose_11/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_11/stack/3?
conv2d_transpose_11/stackPack*conv2d_transpose_11/strided_slice:output:0$conv2d_transpose_11/stack/1:output:0$conv2d_transpose_11/stack/2:output:0$conv2d_transpose_11/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_11/stack?
)conv2d_transpose_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_11/strided_slice_1/stack?
+conv2d_transpose_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_11/strided_slice_1/stack_1?
+conv2d_transpose_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_11/strided_slice_1/stack_2?
#conv2d_transpose_11/strided_slice_1StridedSlice"conv2d_transpose_11/stack:output:02conv2d_transpose_11/strided_slice_1/stack:output:04conv2d_transpose_11/strided_slice_1/stack_1:output:04conv2d_transpose_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_11/strided_slice_1?
3conv2d_transpose_11/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_11_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype025
3conv2d_transpose_11/conv2d_transpose/ReadVariableOp?
)conv2d_transpose_11/conv2d_transpose/CastCast;conv2d_transpose_11/conv2d_transpose/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: @2+
)conv2d_transpose_11/conv2d_transpose/Cast?
$conv2d_transpose_11/conv2d_transposeConv2DBackpropInput"conv2d_transpose_11/stack:output:0-conv2d_transpose_11/conv2d_transpose/Cast:y:0 activation_53/Relu:activations:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2&
$conv2d_transpose_11/conv2d_transpose?
*conv2d_transpose_11/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*conv2d_transpose_11/BiasAdd/ReadVariableOp?
 conv2d_transpose_11/BiasAdd/CastCast2conv2d_transpose_11/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 conv2d_transpose_11/BiasAdd/Cast?
conv2d_transpose_11/BiasAddBiasAdd-conv2d_transpose_11/conv2d_transpose:output:0$conv2d_transpose_11/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:??????????? 2
conv2d_transpose_11/BiasAddz
concatenate_11/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_11/concat/axis?
concatenate_11/concatConcatV2$conv2d_transpose_11/BiasAdd:output:0 activation_43/Relu:activations:0#concatenate_11/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????@2
concatenate_11/concatw
dropout_23/dropout/ConstConst*
_output_shapes
: *
dtype0*
value
B j?x2
dropout_23/dropout/Const?
dropout_23/dropout/MulMulconcatenate_11/concat:output:0!dropout_23/dropout/Const:output:0*
T0*1
_output_shapes
:???????????@2
dropout_23/dropout/Mul?
dropout_23/dropout/ShapeShapeconcatenate_11/concat:output:0*
T0*
_output_shapes
:2
dropout_23/dropout/Shape?
/dropout_23/dropout/random_uniform/RandomUniformRandomUniform!dropout_23/dropout/Shape:output:0*
T0*1
_output_shapes
:???????????@*
dtype021
/dropout_23/dropout/random_uniform/RandomUniform?
!dropout_23/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value
B j?\2#
!dropout_23/dropout/GreaterEqual/y?
dropout_23/dropout/GreaterEqualGreaterEqual8dropout_23/dropout/random_uniform/RandomUniform:output:0*dropout_23/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:???????????@2!
dropout_23/dropout/GreaterEqual?
dropout_23/dropout/CastCast#dropout_23/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:???????????@2
dropout_23/dropout/Cast?
dropout_23/dropout/Mul_1Muldropout_23/dropout/Mul:z:0dropout_23/dropout/Cast:y:0*
T0*1
_output_shapes
:???????????@2
dropout_23/dropout/Mul_1?
conv2d_58/Conv2D/ReadVariableOpReadVariableOp(conv2d_58_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02!
conv2d_58/Conv2D/ReadVariableOp?
conv2d_58/Conv2D/CastCast'conv2d_58/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:@ 2
conv2d_58/Conv2D/Cast?
conv2d_58/Conv2DConv2Ddropout_23/dropout/Mul_1:z:0conv2d_58/Conv2D/Cast:y:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
conv2d_58/Conv2D?
 conv2d_58/BiasAdd/ReadVariableOpReadVariableOp)conv2d_58_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_58/BiasAdd/ReadVariableOp?
conv2d_58/BiasAdd/CastCast(conv2d_58/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: 2
conv2d_58/BiasAdd/Cast?
conv2d_58/BiasAddBiasAddconv2d_58/Conv2D:output:0conv2d_58/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:??????????? 2
conv2d_58/BiasAdd?
%batch_normalization_55/ReadVariableOpReadVariableOp.batch_normalization_55_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_55/ReadVariableOp?
'batch_normalization_55/ReadVariableOp_1ReadVariableOp0batch_normalization_55_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_55/ReadVariableOp_1?
6batch_normalization_55/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_55_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_55/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_55_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_55/FusedBatchNormV3FusedBatchNormV3conv2d_58/BiasAdd:output:0-batch_normalization_55/ReadVariableOp:value:0/batch_normalization_55/ReadVariableOp_1:value:0>batch_normalization_55/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_55/FusedBatchNormV3?
%batch_normalization_55/AssignNewValueAssignVariableOp?batch_normalization_55_fusedbatchnormv3_readvariableop_resource4batch_normalization_55/FusedBatchNormV3:batch_mean:07^batch_normalization_55/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*R
_classH
FDloc:@batch_normalization_55/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_55/AssignNewValue?
'batch_normalization_55/AssignNewValue_1AssignVariableOpAbatch_normalization_55_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_55/FusedBatchNormV3:batch_variance:09^batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*T
_classJ
HFloc:@batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_55/AssignNewValue_1?
activation_55/ReluRelu+batch_normalization_55/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? 2
activation_55/Relu?
conv2d_59/Conv2D/ReadVariableOpReadVariableOp(conv2d_59_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_59/Conv2D/ReadVariableOp?
conv2d_59/Conv2D/CastCast'conv2d_59/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: 2
conv2d_59/Conv2D/Cast?
conv2d_59/Conv2DConv2D activation_55/Relu:activations:0conv2d_59/Conv2D/Cast:y:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
conv2d_59/Conv2D?
 conv2d_59/BiasAdd/ReadVariableOpReadVariableOp)conv2d_59_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_59/BiasAdd/ReadVariableOp?
conv2d_59/BiasAdd/CastCast(conv2d_59/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:2
conv2d_59/BiasAdd/Cast?
conv2d_59/BiasAddBiasAddconv2d_59/Conv2D:output:0conv2d_59/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:???????????2
conv2d_59/BiasAdd?
conv2d_59/SigmoidSigmoidconv2d_59/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_59/Sigmoid?
IdentityIdentityconv2d_59/Sigmoid:y:0&^batch_normalization_43/AssignNewValue(^batch_normalization_43/AssignNewValue_17^batch_normalization_43/FusedBatchNormV3/ReadVariableOp9^batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_43/ReadVariableOp(^batch_normalization_43/ReadVariableOp_1&^batch_normalization_45/AssignNewValue(^batch_normalization_45/AssignNewValue_17^batch_normalization_45/FusedBatchNormV3/ReadVariableOp9^batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_45/ReadVariableOp(^batch_normalization_45/ReadVariableOp_1&^batch_normalization_47/AssignNewValue(^batch_normalization_47/AssignNewValue_17^batch_normalization_47/FusedBatchNormV3/ReadVariableOp9^batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_47/ReadVariableOp(^batch_normalization_47/ReadVariableOp_1&^batch_normalization_49/AssignNewValue(^batch_normalization_49/AssignNewValue_17^batch_normalization_49/FusedBatchNormV3/ReadVariableOp9^batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_49/ReadVariableOp(^batch_normalization_49/ReadVariableOp_1&^batch_normalization_51/AssignNewValue(^batch_normalization_51/AssignNewValue_17^batch_normalization_51/FusedBatchNormV3/ReadVariableOp9^batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_51/ReadVariableOp(^batch_normalization_51/ReadVariableOp_1&^batch_normalization_53/AssignNewValue(^batch_normalization_53/AssignNewValue_17^batch_normalization_53/FusedBatchNormV3/ReadVariableOp9^batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_53/ReadVariableOp(^batch_normalization_53/ReadVariableOp_1&^batch_normalization_55/AssignNewValue(^batch_normalization_55/AssignNewValue_17^batch_normalization_55/FusedBatchNormV3/ReadVariableOp9^batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_55/ReadVariableOp(^batch_normalization_55/ReadVariableOp_1!^conv2d_46/BiasAdd/ReadVariableOp ^conv2d_46/Conv2D/ReadVariableOp!^conv2d_48/BiasAdd/ReadVariableOp ^conv2d_48/Conv2D/ReadVariableOp!^conv2d_50/BiasAdd/ReadVariableOp ^conv2d_50/Conv2D/ReadVariableOp!^conv2d_52/BiasAdd/ReadVariableOp ^conv2d_52/Conv2D/ReadVariableOp!^conv2d_54/BiasAdd/ReadVariableOp ^conv2d_54/Conv2D/ReadVariableOp!^conv2d_56/BiasAdd/ReadVariableOp ^conv2d_56/Conv2D/ReadVariableOp!^conv2d_58/BiasAdd/ReadVariableOp ^conv2d_58/Conv2D/ReadVariableOp!^conv2d_59/BiasAdd/ReadVariableOp ^conv2d_59/Conv2D/ReadVariableOp+^conv2d_transpose_10/BiasAdd/ReadVariableOp4^conv2d_transpose_10/conv2d_transpose/ReadVariableOp+^conv2d_transpose_11/BiasAdd/ReadVariableOp4^conv2d_transpose_11/conv2d_transpose/ReadVariableOp*^conv2d_transpose_9/BiasAdd/ReadVariableOp3^conv2d_transpose_9/conv2d_transpose/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::::::::::2N
%batch_normalization_43/AssignNewValue%batch_normalization_43/AssignNewValue2R
'batch_normalization_43/AssignNewValue_1'batch_normalization_43/AssignNewValue_12p
6batch_normalization_43/FusedBatchNormV3/ReadVariableOp6batch_normalization_43/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_43/FusedBatchNormV3/ReadVariableOp_18batch_normalization_43/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_43/ReadVariableOp%batch_normalization_43/ReadVariableOp2R
'batch_normalization_43/ReadVariableOp_1'batch_normalization_43/ReadVariableOp_12N
%batch_normalization_45/AssignNewValue%batch_normalization_45/AssignNewValue2R
'batch_normalization_45/AssignNewValue_1'batch_normalization_45/AssignNewValue_12p
6batch_normalization_45/FusedBatchNormV3/ReadVariableOp6batch_normalization_45/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_45/FusedBatchNormV3/ReadVariableOp_18batch_normalization_45/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_45/ReadVariableOp%batch_normalization_45/ReadVariableOp2R
'batch_normalization_45/ReadVariableOp_1'batch_normalization_45/ReadVariableOp_12N
%batch_normalization_47/AssignNewValue%batch_normalization_47/AssignNewValue2R
'batch_normalization_47/AssignNewValue_1'batch_normalization_47/AssignNewValue_12p
6batch_normalization_47/FusedBatchNormV3/ReadVariableOp6batch_normalization_47/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_47/FusedBatchNormV3/ReadVariableOp_18batch_normalization_47/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_47/ReadVariableOp%batch_normalization_47/ReadVariableOp2R
'batch_normalization_47/ReadVariableOp_1'batch_normalization_47/ReadVariableOp_12N
%batch_normalization_49/AssignNewValue%batch_normalization_49/AssignNewValue2R
'batch_normalization_49/AssignNewValue_1'batch_normalization_49/AssignNewValue_12p
6batch_normalization_49/FusedBatchNormV3/ReadVariableOp6batch_normalization_49/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_49/FusedBatchNormV3/ReadVariableOp_18batch_normalization_49/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_49/ReadVariableOp%batch_normalization_49/ReadVariableOp2R
'batch_normalization_49/ReadVariableOp_1'batch_normalization_49/ReadVariableOp_12N
%batch_normalization_51/AssignNewValue%batch_normalization_51/AssignNewValue2R
'batch_normalization_51/AssignNewValue_1'batch_normalization_51/AssignNewValue_12p
6batch_normalization_51/FusedBatchNormV3/ReadVariableOp6batch_normalization_51/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_51/FusedBatchNormV3/ReadVariableOp_18batch_normalization_51/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_51/ReadVariableOp%batch_normalization_51/ReadVariableOp2R
'batch_normalization_51/ReadVariableOp_1'batch_normalization_51/ReadVariableOp_12N
%batch_normalization_53/AssignNewValue%batch_normalization_53/AssignNewValue2R
'batch_normalization_53/AssignNewValue_1'batch_normalization_53/AssignNewValue_12p
6batch_normalization_53/FusedBatchNormV3/ReadVariableOp6batch_normalization_53/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_53/FusedBatchNormV3/ReadVariableOp_18batch_normalization_53/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_53/ReadVariableOp%batch_normalization_53/ReadVariableOp2R
'batch_normalization_53/ReadVariableOp_1'batch_normalization_53/ReadVariableOp_12N
%batch_normalization_55/AssignNewValue%batch_normalization_55/AssignNewValue2R
'batch_normalization_55/AssignNewValue_1'batch_normalization_55/AssignNewValue_12p
6batch_normalization_55/FusedBatchNormV3/ReadVariableOp6batch_normalization_55/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_55/FusedBatchNormV3/ReadVariableOp_18batch_normalization_55/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_55/ReadVariableOp%batch_normalization_55/ReadVariableOp2R
'batch_normalization_55/ReadVariableOp_1'batch_normalization_55/ReadVariableOp_12D
 conv2d_46/BiasAdd/ReadVariableOp conv2d_46/BiasAdd/ReadVariableOp2B
conv2d_46/Conv2D/ReadVariableOpconv2d_46/Conv2D/ReadVariableOp2D
 conv2d_48/BiasAdd/ReadVariableOp conv2d_48/BiasAdd/ReadVariableOp2B
conv2d_48/Conv2D/ReadVariableOpconv2d_48/Conv2D/ReadVariableOp2D
 conv2d_50/BiasAdd/ReadVariableOp conv2d_50/BiasAdd/ReadVariableOp2B
conv2d_50/Conv2D/ReadVariableOpconv2d_50/Conv2D/ReadVariableOp2D
 conv2d_52/BiasAdd/ReadVariableOp conv2d_52/BiasAdd/ReadVariableOp2B
conv2d_52/Conv2D/ReadVariableOpconv2d_52/Conv2D/ReadVariableOp2D
 conv2d_54/BiasAdd/ReadVariableOp conv2d_54/BiasAdd/ReadVariableOp2B
conv2d_54/Conv2D/ReadVariableOpconv2d_54/Conv2D/ReadVariableOp2D
 conv2d_56/BiasAdd/ReadVariableOp conv2d_56/BiasAdd/ReadVariableOp2B
conv2d_56/Conv2D/ReadVariableOpconv2d_56/Conv2D/ReadVariableOp2D
 conv2d_58/BiasAdd/ReadVariableOp conv2d_58/BiasAdd/ReadVariableOp2B
conv2d_58/Conv2D/ReadVariableOpconv2d_58/Conv2D/ReadVariableOp2D
 conv2d_59/BiasAdd/ReadVariableOp conv2d_59/BiasAdd/ReadVariableOp2B
conv2d_59/Conv2D/ReadVariableOpconv2d_59/Conv2D/ReadVariableOp2X
*conv2d_transpose_10/BiasAdd/ReadVariableOp*conv2d_transpose_10/BiasAdd/ReadVariableOp2j
3conv2d_transpose_10/conv2d_transpose/ReadVariableOp3conv2d_transpose_10/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_11/BiasAdd/ReadVariableOp*conv2d_transpose_11/BiasAdd/ReadVariableOp2j
3conv2d_transpose_11/conv2d_transpose/ReadVariableOp3conv2d_transpose_11/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_9/BiasAdd/ReadVariableOp)conv2d_transpose_9/BiasAdd/ReadVariableOp2h
2conv2d_transpose_9/conv2d_transpose/ReadVariableOp2conv2d_transpose_9/conv2d_transpose/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
'__inference_model_3_layer_call_fn_79830

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
B__inference_model_3_layer_call_and_return_conditional_losses_788402
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
?
Z
.__inference_concatenate_11_layer_call_fn_80958
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
  ?E8? *R
fMRK
I__inference_concatenate_11_layer_call_and_return_conditional_losses_781232
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
??
?.
 __inference__wrapped_model_76325
input_44
0model_3_conv2d_46_conv2d_readvariableop_resource5
1model_3_conv2d_46_biasadd_readvariableop_resource:
6model_3_batch_normalization_43_readvariableop_resource<
8model_3_batch_normalization_43_readvariableop_1_resourceK
Gmodel_3_batch_normalization_43_fusedbatchnormv3_readvariableop_resourceM
Imodel_3_batch_normalization_43_fusedbatchnormv3_readvariableop_1_resource4
0model_3_conv2d_48_conv2d_readvariableop_resource5
1model_3_conv2d_48_biasadd_readvariableop_resource:
6model_3_batch_normalization_45_readvariableop_resource<
8model_3_batch_normalization_45_readvariableop_1_resourceK
Gmodel_3_batch_normalization_45_fusedbatchnormv3_readvariableop_resourceM
Imodel_3_batch_normalization_45_fusedbatchnormv3_readvariableop_1_resource4
0model_3_conv2d_50_conv2d_readvariableop_resource5
1model_3_conv2d_50_biasadd_readvariableop_resource:
6model_3_batch_normalization_47_readvariableop_resource<
8model_3_batch_normalization_47_readvariableop_1_resourceK
Gmodel_3_batch_normalization_47_fusedbatchnormv3_readvariableop_resourceM
Imodel_3_batch_normalization_47_fusedbatchnormv3_readvariableop_1_resource4
0model_3_conv2d_52_conv2d_readvariableop_resource5
1model_3_conv2d_52_biasadd_readvariableop_resource:
6model_3_batch_normalization_49_readvariableop_resource<
8model_3_batch_normalization_49_readvariableop_1_resourceK
Gmodel_3_batch_normalization_49_fusedbatchnormv3_readvariableop_resourceM
Imodel_3_batch_normalization_49_fusedbatchnormv3_readvariableop_1_resourceG
Cmodel_3_conv2d_transpose_9_conv2d_transpose_readvariableop_resource>
:model_3_conv2d_transpose_9_biasadd_readvariableop_resource4
0model_3_conv2d_54_conv2d_readvariableop_resource5
1model_3_conv2d_54_biasadd_readvariableop_resource:
6model_3_batch_normalization_51_readvariableop_resource<
8model_3_batch_normalization_51_readvariableop_1_resourceK
Gmodel_3_batch_normalization_51_fusedbatchnormv3_readvariableop_resourceM
Imodel_3_batch_normalization_51_fusedbatchnormv3_readvariableop_1_resourceH
Dmodel_3_conv2d_transpose_10_conv2d_transpose_readvariableop_resource?
;model_3_conv2d_transpose_10_biasadd_readvariableop_resource4
0model_3_conv2d_56_conv2d_readvariableop_resource5
1model_3_conv2d_56_biasadd_readvariableop_resource:
6model_3_batch_normalization_53_readvariableop_resource<
8model_3_batch_normalization_53_readvariableop_1_resourceK
Gmodel_3_batch_normalization_53_fusedbatchnormv3_readvariableop_resourceM
Imodel_3_batch_normalization_53_fusedbatchnormv3_readvariableop_1_resourceH
Dmodel_3_conv2d_transpose_11_conv2d_transpose_readvariableop_resource?
;model_3_conv2d_transpose_11_biasadd_readvariableop_resource4
0model_3_conv2d_58_conv2d_readvariableop_resource5
1model_3_conv2d_58_biasadd_readvariableop_resource:
6model_3_batch_normalization_55_readvariableop_resource<
8model_3_batch_normalization_55_readvariableop_1_resourceK
Gmodel_3_batch_normalization_55_fusedbatchnormv3_readvariableop_resourceM
Imodel_3_batch_normalization_55_fusedbatchnormv3_readvariableop_1_resource4
0model_3_conv2d_59_conv2d_readvariableop_resource5
1model_3_conv2d_59_biasadd_readvariableop_resource
identity??>model_3/batch_normalization_43/FusedBatchNormV3/ReadVariableOp?@model_3/batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1?-model_3/batch_normalization_43/ReadVariableOp?/model_3/batch_normalization_43/ReadVariableOp_1?>model_3/batch_normalization_45/FusedBatchNormV3/ReadVariableOp?@model_3/batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1?-model_3/batch_normalization_45/ReadVariableOp?/model_3/batch_normalization_45/ReadVariableOp_1?>model_3/batch_normalization_47/FusedBatchNormV3/ReadVariableOp?@model_3/batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1?-model_3/batch_normalization_47/ReadVariableOp?/model_3/batch_normalization_47/ReadVariableOp_1?>model_3/batch_normalization_49/FusedBatchNormV3/ReadVariableOp?@model_3/batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1?-model_3/batch_normalization_49/ReadVariableOp?/model_3/batch_normalization_49/ReadVariableOp_1?>model_3/batch_normalization_51/FusedBatchNormV3/ReadVariableOp?@model_3/batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1?-model_3/batch_normalization_51/ReadVariableOp?/model_3/batch_normalization_51/ReadVariableOp_1?>model_3/batch_normalization_53/FusedBatchNormV3/ReadVariableOp?@model_3/batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1?-model_3/batch_normalization_53/ReadVariableOp?/model_3/batch_normalization_53/ReadVariableOp_1?>model_3/batch_normalization_55/FusedBatchNormV3/ReadVariableOp?@model_3/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1?-model_3/batch_normalization_55/ReadVariableOp?/model_3/batch_normalization_55/ReadVariableOp_1?(model_3/conv2d_46/BiasAdd/ReadVariableOp?'model_3/conv2d_46/Conv2D/ReadVariableOp?(model_3/conv2d_48/BiasAdd/ReadVariableOp?'model_3/conv2d_48/Conv2D/ReadVariableOp?(model_3/conv2d_50/BiasAdd/ReadVariableOp?'model_3/conv2d_50/Conv2D/ReadVariableOp?(model_3/conv2d_52/BiasAdd/ReadVariableOp?'model_3/conv2d_52/Conv2D/ReadVariableOp?(model_3/conv2d_54/BiasAdd/ReadVariableOp?'model_3/conv2d_54/Conv2D/ReadVariableOp?(model_3/conv2d_56/BiasAdd/ReadVariableOp?'model_3/conv2d_56/Conv2D/ReadVariableOp?(model_3/conv2d_58/BiasAdd/ReadVariableOp?'model_3/conv2d_58/Conv2D/ReadVariableOp?(model_3/conv2d_59/BiasAdd/ReadVariableOp?'model_3/conv2d_59/Conv2D/ReadVariableOp?2model_3/conv2d_transpose_10/BiasAdd/ReadVariableOp?;model_3/conv2d_transpose_10/conv2d_transpose/ReadVariableOp?2model_3/conv2d_transpose_11/BiasAdd/ReadVariableOp?;model_3/conv2d_transpose_11/conv2d_transpose/ReadVariableOp?1model_3/conv2d_transpose_9/BiasAdd/ReadVariableOp?:model_3/conv2d_transpose_9/conv2d_transpose/ReadVariableOpx
model_3/CastCastinput_4*

DstT0*

SrcT0*1
_output_shapes
:???????????2
model_3/Cast?
'model_3/conv2d_46/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_46_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'model_3/conv2d_46/Conv2D/ReadVariableOp?
model_3/conv2d_46/Conv2D/CastCast/model_3/conv2d_46/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: 2
model_3/conv2d_46/Conv2D/Cast?
model_3/conv2d_46/Conv2DConv2Dmodel_3/Cast:y:0!model_3/conv2d_46/Conv2D/Cast:y:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
model_3/conv2d_46/Conv2D?
(model_3/conv2d_46/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_46_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model_3/conv2d_46/BiasAdd/ReadVariableOp?
model_3/conv2d_46/BiasAdd/CastCast0model_3/conv2d_46/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: 2 
model_3/conv2d_46/BiasAdd/Cast?
model_3/conv2d_46/BiasAddBiasAdd!model_3/conv2d_46/Conv2D:output:0"model_3/conv2d_46/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:??????????? 2
model_3/conv2d_46/BiasAdd?
-model_3/batch_normalization_43/ReadVariableOpReadVariableOp6model_3_batch_normalization_43_readvariableop_resource*
_output_shapes
: *
dtype02/
-model_3/batch_normalization_43/ReadVariableOp?
/model_3/batch_normalization_43/ReadVariableOp_1ReadVariableOp8model_3_batch_normalization_43_readvariableop_1_resource*
_output_shapes
: *
dtype021
/model_3/batch_normalization_43/ReadVariableOp_1?
>model_3/batch_normalization_43/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_3_batch_normalization_43_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02@
>model_3/batch_normalization_43/FusedBatchNormV3/ReadVariableOp?
@model_3/batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_3_batch_normalization_43_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02B
@model_3/batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1?
/model_3/batch_normalization_43/FusedBatchNormV3FusedBatchNormV3"model_3/conv2d_46/BiasAdd:output:05model_3/batch_normalization_43/ReadVariableOp:value:07model_3/batch_normalization_43/ReadVariableOp_1:value:0Fmodel_3/batch_normalization_43/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_3/batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 21
/model_3/batch_normalization_43/FusedBatchNormV3?
model_3/activation_43/ReluRelu3model_3/batch_normalization_43/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? 2
model_3/activation_43/Relu?
model_3/max_pooling2d_9/MaxPoolMaxPool(model_3/activation_43/Relu:activations:0*
T0*1
_output_shapes
:??????????? *
ksize
*
paddingVALID*
strides
2!
model_3/max_pooling2d_9/MaxPool?
model_3/dropout_18/IdentityIdentity(model_3/max_pooling2d_9/MaxPool:output:0*
T0*1
_output_shapes
:??????????? 2
model_3/dropout_18/Identity?
'model_3/conv2d_48/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_48_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02)
'model_3/conv2d_48/Conv2D/ReadVariableOp?
model_3/conv2d_48/Conv2D/CastCast/model_3/conv2d_48/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: @2
model_3/conv2d_48/Conv2D/Cast?
model_3/conv2d_48/Conv2DConv2D$model_3/dropout_18/Identity:output:0!model_3/conv2d_48/Conv2D/Cast:y:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
model_3/conv2d_48/Conv2D?
(model_3/conv2d_48/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_48_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_3/conv2d_48/BiasAdd/ReadVariableOp?
model_3/conv2d_48/BiasAdd/CastCast0model_3/conv2d_48/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@2 
model_3/conv2d_48/BiasAdd/Cast?
model_3/conv2d_48/BiasAddBiasAdd!model_3/conv2d_48/Conv2D:output:0"model_3/conv2d_48/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:???????????@2
model_3/conv2d_48/BiasAdd?
-model_3/batch_normalization_45/ReadVariableOpReadVariableOp6model_3_batch_normalization_45_readvariableop_resource*
_output_shapes
:@*
dtype02/
-model_3/batch_normalization_45/ReadVariableOp?
/model_3/batch_normalization_45/ReadVariableOp_1ReadVariableOp8model_3_batch_normalization_45_readvariableop_1_resource*
_output_shapes
:@*
dtype021
/model_3/batch_normalization_45/ReadVariableOp_1?
>model_3/batch_normalization_45/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_3_batch_normalization_45_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02@
>model_3/batch_normalization_45/FusedBatchNormV3/ReadVariableOp?
@model_3/batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_3_batch_normalization_45_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02B
@model_3/batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1?
/model_3/batch_normalization_45/FusedBatchNormV3FusedBatchNormV3"model_3/conv2d_48/BiasAdd:output:05model_3/batch_normalization_45/ReadVariableOp:value:07model_3/batch_normalization_45/ReadVariableOp_1:value:0Fmodel_3/batch_normalization_45/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_3/batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
is_training( 21
/model_3/batch_normalization_45/FusedBatchNormV3?
model_3/activation_45/ReluRelu3model_3/batch_normalization_45/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????@2
model_3/activation_45/Relu?
 model_3/max_pooling2d_10/MaxPoolMaxPool(model_3/activation_45/Relu:activations:0*
T0*1
_output_shapes
:???????????@*
ksize
*
paddingVALID*
strides
2"
 model_3/max_pooling2d_10/MaxPool?
model_3/dropout_19/IdentityIdentity)model_3/max_pooling2d_10/MaxPool:output:0*
T0*1
_output_shapes
:???????????@2
model_3/dropout_19/Identity?
'model_3/conv2d_50/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_50_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02)
'model_3/conv2d_50/Conv2D/ReadVariableOp?
model_3/conv2d_50/Conv2D/CastCast/model_3/conv2d_50/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*'
_output_shapes
:@?2
model_3/conv2d_50/Conv2D/Cast?
model_3/conv2d_50/Conv2DConv2D$model_3/dropout_19/Identity:output:0!model_3/conv2d_50/Conv2D/Cast:y:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
model_3/conv2d_50/Conv2D?
(model_3/conv2d_50/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_50_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(model_3/conv2d_50/BiasAdd/ReadVariableOp?
model_3/conv2d_50/BiasAdd/CastCast0model_3/conv2d_50/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:?2 
model_3/conv2d_50/BiasAdd/Cast?
model_3/conv2d_50/BiasAddBiasAdd!model_3/conv2d_50/Conv2D:output:0"model_3/conv2d_50/BiasAdd/Cast:y:0*
T0*2
_output_shapes 
:????????????2
model_3/conv2d_50/BiasAdd?
-model_3/batch_normalization_47/ReadVariableOpReadVariableOp6model_3_batch_normalization_47_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-model_3/batch_normalization_47/ReadVariableOp?
/model_3/batch_normalization_47/ReadVariableOp_1ReadVariableOp8model_3_batch_normalization_47_readvariableop_1_resource*
_output_shapes	
:?*
dtype021
/model_3/batch_normalization_47/ReadVariableOp_1?
>model_3/batch_normalization_47/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_3_batch_normalization_47_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02@
>model_3/batch_normalization_47/FusedBatchNormV3/ReadVariableOp?
@model_3/batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_3_batch_normalization_47_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02B
@model_3/batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1?
/model_3/batch_normalization_47/FusedBatchNormV3FusedBatchNormV3"model_3/conv2d_50/BiasAdd:output:05model_3/batch_normalization_47/ReadVariableOp:value:07model_3/batch_normalization_47/ReadVariableOp_1:value:0Fmodel_3/batch_normalization_47/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_3/batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:????????????:?:?:?:?:*
epsilon%o?:*
is_training( 21
/model_3/batch_normalization_47/FusedBatchNormV3?
model_3/activation_47/ReluRelu3model_3/batch_normalization_47/FusedBatchNormV3:y:0*
T0*2
_output_shapes 
:????????????2
model_3/activation_47/Relu?
 model_3/max_pooling2d_11/MaxPoolMaxPool(model_3/activation_47/Relu:activations:0*
T0*0
_output_shapes
:?????????D\?*
ksize
*
paddingVALID*
strides
2"
 model_3/max_pooling2d_11/MaxPool?
model_3/dropout_20/IdentityIdentity)model_3/max_pooling2d_11/MaxPool:output:0*
T0*0
_output_shapes
:?????????D\?2
model_3/dropout_20/Identity?
'model_3/conv2d_52/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_52_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02)
'model_3/conv2d_52/Conv2D/ReadVariableOp?
model_3/conv2d_52/Conv2D/CastCast/model_3/conv2d_52/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*(
_output_shapes
:??2
model_3/conv2d_52/Conv2D/Cast?
model_3/conv2d_52/Conv2DConv2D$model_3/dropout_20/Identity:output:0!model_3/conv2d_52/Conv2D/Cast:y:0*
T0*0
_output_shapes
:?????????D\?*
paddingSAME*
strides
2
model_3/conv2d_52/Conv2D?
(model_3/conv2d_52/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_52_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(model_3/conv2d_52/BiasAdd/ReadVariableOp?
model_3/conv2d_52/BiasAdd/CastCast0model_3/conv2d_52/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:?2 
model_3/conv2d_52/BiasAdd/Cast?
model_3/conv2d_52/BiasAddBiasAdd!model_3/conv2d_52/Conv2D:output:0"model_3/conv2d_52/BiasAdd/Cast:y:0*
T0*0
_output_shapes
:?????????D\?2
model_3/conv2d_52/BiasAdd?
-model_3/batch_normalization_49/ReadVariableOpReadVariableOp6model_3_batch_normalization_49_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-model_3/batch_normalization_49/ReadVariableOp?
/model_3/batch_normalization_49/ReadVariableOp_1ReadVariableOp8model_3_batch_normalization_49_readvariableop_1_resource*
_output_shapes	
:?*
dtype021
/model_3/batch_normalization_49/ReadVariableOp_1?
>model_3/batch_normalization_49/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_3_batch_normalization_49_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02@
>model_3/batch_normalization_49/FusedBatchNormV3/ReadVariableOp?
@model_3/batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_3_batch_normalization_49_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02B
@model_3/batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1?
/model_3/batch_normalization_49/FusedBatchNormV3FusedBatchNormV3"model_3/conv2d_52/BiasAdd:output:05model_3/batch_normalization_49/ReadVariableOp:value:07model_3/batch_normalization_49/ReadVariableOp_1:value:0Fmodel_3/batch_normalization_49/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_3/batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????D\?:?:?:?:?:*
epsilon%o?:*
is_training( 21
/model_3/batch_normalization_49/FusedBatchNormV3?
model_3/activation_49/ReluRelu3model_3/batch_normalization_49/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????D\?2
model_3/activation_49/Relu?
 model_3/conv2d_transpose_9/ShapeShape(model_3/activation_49/Relu:activations:0*
T0*
_output_shapes
:2"
 model_3/conv2d_transpose_9/Shape?
.model_3/conv2d_transpose_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.model_3/conv2d_transpose_9/strided_slice/stack?
0model_3/conv2d_transpose_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0model_3/conv2d_transpose_9/strided_slice/stack_1?
0model_3/conv2d_transpose_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0model_3/conv2d_transpose_9/strided_slice/stack_2?
(model_3/conv2d_transpose_9/strided_sliceStridedSlice)model_3/conv2d_transpose_9/Shape:output:07model_3/conv2d_transpose_9/strided_slice/stack:output:09model_3/conv2d_transpose_9/strided_slice/stack_1:output:09model_3/conv2d_transpose_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(model_3/conv2d_transpose_9/strided_slice?
"model_3/conv2d_transpose_9/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2$
"model_3/conv2d_transpose_9/stack/1?
"model_3/conv2d_transpose_9/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2$
"model_3/conv2d_transpose_9/stack/2?
"model_3/conv2d_transpose_9/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2$
"model_3/conv2d_transpose_9/stack/3?
 model_3/conv2d_transpose_9/stackPack1model_3/conv2d_transpose_9/strided_slice:output:0+model_3/conv2d_transpose_9/stack/1:output:0+model_3/conv2d_transpose_9/stack/2:output:0+model_3/conv2d_transpose_9/stack/3:output:0*
N*
T0*
_output_shapes
:2"
 model_3/conv2d_transpose_9/stack?
0model_3/conv2d_transpose_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0model_3/conv2d_transpose_9/strided_slice_1/stack?
2model_3/conv2d_transpose_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2model_3/conv2d_transpose_9/strided_slice_1/stack_1?
2model_3/conv2d_transpose_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2model_3/conv2d_transpose_9/strided_slice_1/stack_2?
*model_3/conv2d_transpose_9/strided_slice_1StridedSlice)model_3/conv2d_transpose_9/stack:output:09model_3/conv2d_transpose_9/strided_slice_1/stack:output:0;model_3/conv2d_transpose_9/strided_slice_1/stack_1:output:0;model_3/conv2d_transpose_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*model_3/conv2d_transpose_9/strided_slice_1?
:model_3/conv2d_transpose_9/conv2d_transpose/ReadVariableOpReadVariableOpCmodel_3_conv2d_transpose_9_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype02<
:model_3/conv2d_transpose_9/conv2d_transpose/ReadVariableOp?
0model_3/conv2d_transpose_9/conv2d_transpose/CastCastBmodel_3/conv2d_transpose_9/conv2d_transpose/ReadVariableOp:value:0*

DstT0*

SrcT0*(
_output_shapes
:??22
0model_3/conv2d_transpose_9/conv2d_transpose/Cast?
+model_3/conv2d_transpose_9/conv2d_transposeConv2DBackpropInput)model_3/conv2d_transpose_9/stack:output:04model_3/conv2d_transpose_9/conv2d_transpose/Cast:y:0(model_3/activation_49/Relu:activations:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2-
+model_3/conv2d_transpose_9/conv2d_transpose?
1model_3/conv2d_transpose_9/BiasAdd/ReadVariableOpReadVariableOp:model_3_conv2d_transpose_9_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype023
1model_3/conv2d_transpose_9/BiasAdd/ReadVariableOp?
'model_3/conv2d_transpose_9/BiasAdd/CastCast9model_3/conv2d_transpose_9/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:?2)
'model_3/conv2d_transpose_9/BiasAdd/Cast?
"model_3/conv2d_transpose_9/BiasAddBiasAdd4model_3/conv2d_transpose_9/conv2d_transpose:output:0+model_3/conv2d_transpose_9/BiasAdd/Cast:y:0*
T0*2
_output_shapes 
:????????????2$
"model_3/conv2d_transpose_9/BiasAdd?
!model_3/concatenate_9/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_3/concatenate_9/concat/axis?
model_3/concatenate_9/concatConcatV2+model_3/conv2d_transpose_9/BiasAdd:output:0(model_3/activation_47/Relu:activations:0*model_3/concatenate_9/concat/axis:output:0*
N*
T0*2
_output_shapes 
:????????????2
model_3/concatenate_9/concat?
model_3/dropout_21/IdentityIdentity%model_3/concatenate_9/concat:output:0*
T0*2
_output_shapes 
:????????????2
model_3/dropout_21/Identity?
'model_3/conv2d_54/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_54_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02)
'model_3/conv2d_54/Conv2D/ReadVariableOp?
model_3/conv2d_54/Conv2D/CastCast/model_3/conv2d_54/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*(
_output_shapes
:??2
model_3/conv2d_54/Conv2D/Cast?
model_3/conv2d_54/Conv2DConv2D$model_3/dropout_21/Identity:output:0!model_3/conv2d_54/Conv2D/Cast:y:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
model_3/conv2d_54/Conv2D?
(model_3/conv2d_54/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_54_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(model_3/conv2d_54/BiasAdd/ReadVariableOp?
model_3/conv2d_54/BiasAdd/CastCast0model_3/conv2d_54/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:?2 
model_3/conv2d_54/BiasAdd/Cast?
model_3/conv2d_54/BiasAddBiasAdd!model_3/conv2d_54/Conv2D:output:0"model_3/conv2d_54/BiasAdd/Cast:y:0*
T0*2
_output_shapes 
:????????????2
model_3/conv2d_54/BiasAdd?
-model_3/batch_normalization_51/ReadVariableOpReadVariableOp6model_3_batch_normalization_51_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-model_3/batch_normalization_51/ReadVariableOp?
/model_3/batch_normalization_51/ReadVariableOp_1ReadVariableOp8model_3_batch_normalization_51_readvariableop_1_resource*
_output_shapes	
:?*
dtype021
/model_3/batch_normalization_51/ReadVariableOp_1?
>model_3/batch_normalization_51/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_3_batch_normalization_51_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02@
>model_3/batch_normalization_51/FusedBatchNormV3/ReadVariableOp?
@model_3/batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_3_batch_normalization_51_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02B
@model_3/batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1?
/model_3/batch_normalization_51/FusedBatchNormV3FusedBatchNormV3"model_3/conv2d_54/BiasAdd:output:05model_3/batch_normalization_51/ReadVariableOp:value:07model_3/batch_normalization_51/ReadVariableOp_1:value:0Fmodel_3/batch_normalization_51/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_3/batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:????????????:?:?:?:?:*
epsilon%o?:*
is_training( 21
/model_3/batch_normalization_51/FusedBatchNormV3?
model_3/activation_51/ReluRelu3model_3/batch_normalization_51/FusedBatchNormV3:y:0*
T0*2
_output_shapes 
:????????????2
model_3/activation_51/Relu?
!model_3/conv2d_transpose_10/ShapeShape(model_3/activation_51/Relu:activations:0*
T0*
_output_shapes
:2#
!model_3/conv2d_transpose_10/Shape?
/model_3/conv2d_transpose_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/model_3/conv2d_transpose_10/strided_slice/stack?
1model_3/conv2d_transpose_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1model_3/conv2d_transpose_10/strided_slice/stack_1?
1model_3/conv2d_transpose_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1model_3/conv2d_transpose_10/strided_slice/stack_2?
)model_3/conv2d_transpose_10/strided_sliceStridedSlice*model_3/conv2d_transpose_10/Shape:output:08model_3/conv2d_transpose_10/strided_slice/stack:output:0:model_3/conv2d_transpose_10/strided_slice/stack_1:output:0:model_3/conv2d_transpose_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)model_3/conv2d_transpose_10/strided_slice?
#model_3/conv2d_transpose_10/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2%
#model_3/conv2d_transpose_10/stack/1?
#model_3/conv2d_transpose_10/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2%
#model_3/conv2d_transpose_10/stack/2?
#model_3/conv2d_transpose_10/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2%
#model_3/conv2d_transpose_10/stack/3?
!model_3/conv2d_transpose_10/stackPack2model_3/conv2d_transpose_10/strided_slice:output:0,model_3/conv2d_transpose_10/stack/1:output:0,model_3/conv2d_transpose_10/stack/2:output:0,model_3/conv2d_transpose_10/stack/3:output:0*
N*
T0*
_output_shapes
:2#
!model_3/conv2d_transpose_10/stack?
1model_3/conv2d_transpose_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1model_3/conv2d_transpose_10/strided_slice_1/stack?
3model_3/conv2d_transpose_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3model_3/conv2d_transpose_10/strided_slice_1/stack_1?
3model_3/conv2d_transpose_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3model_3/conv2d_transpose_10/strided_slice_1/stack_2?
+model_3/conv2d_transpose_10/strided_slice_1StridedSlice*model_3/conv2d_transpose_10/stack:output:0:model_3/conv2d_transpose_10/strided_slice_1/stack:output:0<model_3/conv2d_transpose_10/strided_slice_1/stack_1:output:0<model_3/conv2d_transpose_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+model_3/conv2d_transpose_10/strided_slice_1?
;model_3/conv2d_transpose_10/conv2d_transpose/ReadVariableOpReadVariableOpDmodel_3_conv2d_transpose_10_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype02=
;model_3/conv2d_transpose_10/conv2d_transpose/ReadVariableOp?
1model_3/conv2d_transpose_10/conv2d_transpose/CastCastCmodel_3/conv2d_transpose_10/conv2d_transpose/ReadVariableOp:value:0*

DstT0*

SrcT0*'
_output_shapes
:@?23
1model_3/conv2d_transpose_10/conv2d_transpose/Cast?
,model_3/conv2d_transpose_10/conv2d_transposeConv2DBackpropInput*model_3/conv2d_transpose_10/stack:output:05model_3/conv2d_transpose_10/conv2d_transpose/Cast:y:0(model_3/activation_51/Relu:activations:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2.
,model_3/conv2d_transpose_10/conv2d_transpose?
2model_3/conv2d_transpose_10/BiasAdd/ReadVariableOpReadVariableOp;model_3_conv2d_transpose_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype024
2model_3/conv2d_transpose_10/BiasAdd/ReadVariableOp?
(model_3/conv2d_transpose_10/BiasAdd/CastCast:model_3/conv2d_transpose_10/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@2*
(model_3/conv2d_transpose_10/BiasAdd/Cast?
#model_3/conv2d_transpose_10/BiasAddBiasAdd5model_3/conv2d_transpose_10/conv2d_transpose:output:0,model_3/conv2d_transpose_10/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:???????????@2%
#model_3/conv2d_transpose_10/BiasAdd?
"model_3/concatenate_10/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2$
"model_3/concatenate_10/concat/axis?
model_3/concatenate_10/concatConcatV2,model_3/conv2d_transpose_10/BiasAdd:output:0(model_3/activation_45/Relu:activations:0+model_3/concatenate_10/concat/axis:output:0*
N*
T0*2
_output_shapes 
:????????????2
model_3/concatenate_10/concat?
model_3/dropout_22/IdentityIdentity&model_3/concatenate_10/concat:output:0*
T0*2
_output_shapes 
:????????????2
model_3/dropout_22/Identity?
'model_3/conv2d_56/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_56_conv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02)
'model_3/conv2d_56/Conv2D/ReadVariableOp?
model_3/conv2d_56/Conv2D/CastCast/model_3/conv2d_56/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*'
_output_shapes
:?@2
model_3/conv2d_56/Conv2D/Cast?
model_3/conv2d_56/Conv2DConv2D$model_3/dropout_22/Identity:output:0!model_3/conv2d_56/Conv2D/Cast:y:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
model_3/conv2d_56/Conv2D?
(model_3/conv2d_56/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_56_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_3/conv2d_56/BiasAdd/ReadVariableOp?
model_3/conv2d_56/BiasAdd/CastCast0model_3/conv2d_56/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@2 
model_3/conv2d_56/BiasAdd/Cast?
model_3/conv2d_56/BiasAddBiasAdd!model_3/conv2d_56/Conv2D:output:0"model_3/conv2d_56/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:???????????@2
model_3/conv2d_56/BiasAdd?
-model_3/batch_normalization_53/ReadVariableOpReadVariableOp6model_3_batch_normalization_53_readvariableop_resource*
_output_shapes
:@*
dtype02/
-model_3/batch_normalization_53/ReadVariableOp?
/model_3/batch_normalization_53/ReadVariableOp_1ReadVariableOp8model_3_batch_normalization_53_readvariableop_1_resource*
_output_shapes
:@*
dtype021
/model_3/batch_normalization_53/ReadVariableOp_1?
>model_3/batch_normalization_53/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_3_batch_normalization_53_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02@
>model_3/batch_normalization_53/FusedBatchNormV3/ReadVariableOp?
@model_3/batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_3_batch_normalization_53_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02B
@model_3/batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1?
/model_3/batch_normalization_53/FusedBatchNormV3FusedBatchNormV3"model_3/conv2d_56/BiasAdd:output:05model_3/batch_normalization_53/ReadVariableOp:value:07model_3/batch_normalization_53/ReadVariableOp_1:value:0Fmodel_3/batch_normalization_53/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_3/batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
is_training( 21
/model_3/batch_normalization_53/FusedBatchNormV3?
model_3/activation_53/ReluRelu3model_3/batch_normalization_53/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????@2
model_3/activation_53/Relu?
!model_3/conv2d_transpose_11/ShapeShape(model_3/activation_53/Relu:activations:0*
T0*
_output_shapes
:2#
!model_3/conv2d_transpose_11/Shape?
/model_3/conv2d_transpose_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/model_3/conv2d_transpose_11/strided_slice/stack?
1model_3/conv2d_transpose_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1model_3/conv2d_transpose_11/strided_slice/stack_1?
1model_3/conv2d_transpose_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1model_3/conv2d_transpose_11/strided_slice/stack_2?
)model_3/conv2d_transpose_11/strided_sliceStridedSlice*model_3/conv2d_transpose_11/Shape:output:08model_3/conv2d_transpose_11/strided_slice/stack:output:0:model_3/conv2d_transpose_11/strided_slice/stack_1:output:0:model_3/conv2d_transpose_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)model_3/conv2d_transpose_11/strided_slice?
#model_3/conv2d_transpose_11/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2%
#model_3/conv2d_transpose_11/stack/1?
#model_3/conv2d_transpose_11/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2%
#model_3/conv2d_transpose_11/stack/2?
#model_3/conv2d_transpose_11/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2%
#model_3/conv2d_transpose_11/stack/3?
!model_3/conv2d_transpose_11/stackPack2model_3/conv2d_transpose_11/strided_slice:output:0,model_3/conv2d_transpose_11/stack/1:output:0,model_3/conv2d_transpose_11/stack/2:output:0,model_3/conv2d_transpose_11/stack/3:output:0*
N*
T0*
_output_shapes
:2#
!model_3/conv2d_transpose_11/stack?
1model_3/conv2d_transpose_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1model_3/conv2d_transpose_11/strided_slice_1/stack?
3model_3/conv2d_transpose_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3model_3/conv2d_transpose_11/strided_slice_1/stack_1?
3model_3/conv2d_transpose_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3model_3/conv2d_transpose_11/strided_slice_1/stack_2?
+model_3/conv2d_transpose_11/strided_slice_1StridedSlice*model_3/conv2d_transpose_11/stack:output:0:model_3/conv2d_transpose_11/strided_slice_1/stack:output:0<model_3/conv2d_transpose_11/strided_slice_1/stack_1:output:0<model_3/conv2d_transpose_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+model_3/conv2d_transpose_11/strided_slice_1?
;model_3/conv2d_transpose_11/conv2d_transpose/ReadVariableOpReadVariableOpDmodel_3_conv2d_transpose_11_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02=
;model_3/conv2d_transpose_11/conv2d_transpose/ReadVariableOp?
1model_3/conv2d_transpose_11/conv2d_transpose/CastCastCmodel_3/conv2d_transpose_11/conv2d_transpose/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: @23
1model_3/conv2d_transpose_11/conv2d_transpose/Cast?
,model_3/conv2d_transpose_11/conv2d_transposeConv2DBackpropInput*model_3/conv2d_transpose_11/stack:output:05model_3/conv2d_transpose_11/conv2d_transpose/Cast:y:0(model_3/activation_53/Relu:activations:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2.
,model_3/conv2d_transpose_11/conv2d_transpose?
2model_3/conv2d_transpose_11/BiasAdd/ReadVariableOpReadVariableOp;model_3_conv2d_transpose_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype024
2model_3/conv2d_transpose_11/BiasAdd/ReadVariableOp?
(model_3/conv2d_transpose_11/BiasAdd/CastCast:model_3/conv2d_transpose_11/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: 2*
(model_3/conv2d_transpose_11/BiasAdd/Cast?
#model_3/conv2d_transpose_11/BiasAddBiasAdd5model_3/conv2d_transpose_11/conv2d_transpose:output:0,model_3/conv2d_transpose_11/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:??????????? 2%
#model_3/conv2d_transpose_11/BiasAdd?
"model_3/concatenate_11/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2$
"model_3/concatenate_11/concat/axis?
model_3/concatenate_11/concatConcatV2,model_3/conv2d_transpose_11/BiasAdd:output:0(model_3/activation_43/Relu:activations:0+model_3/concatenate_11/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????@2
model_3/concatenate_11/concat?
model_3/dropout_23/IdentityIdentity&model_3/concatenate_11/concat:output:0*
T0*1
_output_shapes
:???????????@2
model_3/dropout_23/Identity?
'model_3/conv2d_58/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_58_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02)
'model_3/conv2d_58/Conv2D/ReadVariableOp?
model_3/conv2d_58/Conv2D/CastCast/model_3/conv2d_58/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:@ 2
model_3/conv2d_58/Conv2D/Cast?
model_3/conv2d_58/Conv2DConv2D$model_3/dropout_23/Identity:output:0!model_3/conv2d_58/Conv2D/Cast:y:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
model_3/conv2d_58/Conv2D?
(model_3/conv2d_58/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_58_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model_3/conv2d_58/BiasAdd/ReadVariableOp?
model_3/conv2d_58/BiasAdd/CastCast0model_3/conv2d_58/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: 2 
model_3/conv2d_58/BiasAdd/Cast?
model_3/conv2d_58/BiasAddBiasAdd!model_3/conv2d_58/Conv2D:output:0"model_3/conv2d_58/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:??????????? 2
model_3/conv2d_58/BiasAdd?
-model_3/batch_normalization_55/ReadVariableOpReadVariableOp6model_3_batch_normalization_55_readvariableop_resource*
_output_shapes
: *
dtype02/
-model_3/batch_normalization_55/ReadVariableOp?
/model_3/batch_normalization_55/ReadVariableOp_1ReadVariableOp8model_3_batch_normalization_55_readvariableop_1_resource*
_output_shapes
: *
dtype021
/model_3/batch_normalization_55/ReadVariableOp_1?
>model_3/batch_normalization_55/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_3_batch_normalization_55_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02@
>model_3/batch_normalization_55/FusedBatchNormV3/ReadVariableOp?
@model_3/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_3_batch_normalization_55_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02B
@model_3/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1?
/model_3/batch_normalization_55/FusedBatchNormV3FusedBatchNormV3"model_3/conv2d_58/BiasAdd:output:05model_3/batch_normalization_55/ReadVariableOp:value:07model_3/batch_normalization_55/ReadVariableOp_1:value:0Fmodel_3/batch_normalization_55/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_3/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 21
/model_3/batch_normalization_55/FusedBatchNormV3?
model_3/activation_55/ReluRelu3model_3/batch_normalization_55/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? 2
model_3/activation_55/Relu?
'model_3/conv2d_59/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_59_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'model_3/conv2d_59/Conv2D/ReadVariableOp?
model_3/conv2d_59/Conv2D/CastCast/model_3/conv2d_59/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: 2
model_3/conv2d_59/Conv2D/Cast?
model_3/conv2d_59/Conv2DConv2D(model_3/activation_55/Relu:activations:0!model_3/conv2d_59/Conv2D/Cast:y:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
model_3/conv2d_59/Conv2D?
(model_3/conv2d_59/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_59_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_3/conv2d_59/BiasAdd/ReadVariableOp?
model_3/conv2d_59/BiasAdd/CastCast0model_3/conv2d_59/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:2 
model_3/conv2d_59/BiasAdd/Cast?
model_3/conv2d_59/BiasAddBiasAdd!model_3/conv2d_59/Conv2D:output:0"model_3/conv2d_59/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:???????????2
model_3/conv2d_59/BiasAdd?
model_3/conv2d_59/SigmoidSigmoid"model_3/conv2d_59/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
model_3/conv2d_59/Sigmoid?
IdentityIdentitymodel_3/conv2d_59/Sigmoid:y:0?^model_3/batch_normalization_43/FusedBatchNormV3/ReadVariableOpA^model_3/batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1.^model_3/batch_normalization_43/ReadVariableOp0^model_3/batch_normalization_43/ReadVariableOp_1?^model_3/batch_normalization_45/FusedBatchNormV3/ReadVariableOpA^model_3/batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1.^model_3/batch_normalization_45/ReadVariableOp0^model_3/batch_normalization_45/ReadVariableOp_1?^model_3/batch_normalization_47/FusedBatchNormV3/ReadVariableOpA^model_3/batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1.^model_3/batch_normalization_47/ReadVariableOp0^model_3/batch_normalization_47/ReadVariableOp_1?^model_3/batch_normalization_49/FusedBatchNormV3/ReadVariableOpA^model_3/batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1.^model_3/batch_normalization_49/ReadVariableOp0^model_3/batch_normalization_49/ReadVariableOp_1?^model_3/batch_normalization_51/FusedBatchNormV3/ReadVariableOpA^model_3/batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1.^model_3/batch_normalization_51/ReadVariableOp0^model_3/batch_normalization_51/ReadVariableOp_1?^model_3/batch_normalization_53/FusedBatchNormV3/ReadVariableOpA^model_3/batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1.^model_3/batch_normalization_53/ReadVariableOp0^model_3/batch_normalization_53/ReadVariableOp_1?^model_3/batch_normalization_55/FusedBatchNormV3/ReadVariableOpA^model_3/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1.^model_3/batch_normalization_55/ReadVariableOp0^model_3/batch_normalization_55/ReadVariableOp_1)^model_3/conv2d_46/BiasAdd/ReadVariableOp(^model_3/conv2d_46/Conv2D/ReadVariableOp)^model_3/conv2d_48/BiasAdd/ReadVariableOp(^model_3/conv2d_48/Conv2D/ReadVariableOp)^model_3/conv2d_50/BiasAdd/ReadVariableOp(^model_3/conv2d_50/Conv2D/ReadVariableOp)^model_3/conv2d_52/BiasAdd/ReadVariableOp(^model_3/conv2d_52/Conv2D/ReadVariableOp)^model_3/conv2d_54/BiasAdd/ReadVariableOp(^model_3/conv2d_54/Conv2D/ReadVariableOp)^model_3/conv2d_56/BiasAdd/ReadVariableOp(^model_3/conv2d_56/Conv2D/ReadVariableOp)^model_3/conv2d_58/BiasAdd/ReadVariableOp(^model_3/conv2d_58/Conv2D/ReadVariableOp)^model_3/conv2d_59/BiasAdd/ReadVariableOp(^model_3/conv2d_59/Conv2D/ReadVariableOp3^model_3/conv2d_transpose_10/BiasAdd/ReadVariableOp<^model_3/conv2d_transpose_10/conv2d_transpose/ReadVariableOp3^model_3/conv2d_transpose_11/BiasAdd/ReadVariableOp<^model_3/conv2d_transpose_11/conv2d_transpose/ReadVariableOp2^model_3/conv2d_transpose_9/BiasAdd/ReadVariableOp;^model_3/conv2d_transpose_9/conv2d_transpose/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::::::::::2?
>model_3/batch_normalization_43/FusedBatchNormV3/ReadVariableOp>model_3/batch_normalization_43/FusedBatchNormV3/ReadVariableOp2?
@model_3/batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1@model_3/batch_normalization_43/FusedBatchNormV3/ReadVariableOp_12^
-model_3/batch_normalization_43/ReadVariableOp-model_3/batch_normalization_43/ReadVariableOp2b
/model_3/batch_normalization_43/ReadVariableOp_1/model_3/batch_normalization_43/ReadVariableOp_12?
>model_3/batch_normalization_45/FusedBatchNormV3/ReadVariableOp>model_3/batch_normalization_45/FusedBatchNormV3/ReadVariableOp2?
@model_3/batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1@model_3/batch_normalization_45/FusedBatchNormV3/ReadVariableOp_12^
-model_3/batch_normalization_45/ReadVariableOp-model_3/batch_normalization_45/ReadVariableOp2b
/model_3/batch_normalization_45/ReadVariableOp_1/model_3/batch_normalization_45/ReadVariableOp_12?
>model_3/batch_normalization_47/FusedBatchNormV3/ReadVariableOp>model_3/batch_normalization_47/FusedBatchNormV3/ReadVariableOp2?
@model_3/batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1@model_3/batch_normalization_47/FusedBatchNormV3/ReadVariableOp_12^
-model_3/batch_normalization_47/ReadVariableOp-model_3/batch_normalization_47/ReadVariableOp2b
/model_3/batch_normalization_47/ReadVariableOp_1/model_3/batch_normalization_47/ReadVariableOp_12?
>model_3/batch_normalization_49/FusedBatchNormV3/ReadVariableOp>model_3/batch_normalization_49/FusedBatchNormV3/ReadVariableOp2?
@model_3/batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1@model_3/batch_normalization_49/FusedBatchNormV3/ReadVariableOp_12^
-model_3/batch_normalization_49/ReadVariableOp-model_3/batch_normalization_49/ReadVariableOp2b
/model_3/batch_normalization_49/ReadVariableOp_1/model_3/batch_normalization_49/ReadVariableOp_12?
>model_3/batch_normalization_51/FusedBatchNormV3/ReadVariableOp>model_3/batch_normalization_51/FusedBatchNormV3/ReadVariableOp2?
@model_3/batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1@model_3/batch_normalization_51/FusedBatchNormV3/ReadVariableOp_12^
-model_3/batch_normalization_51/ReadVariableOp-model_3/batch_normalization_51/ReadVariableOp2b
/model_3/batch_normalization_51/ReadVariableOp_1/model_3/batch_normalization_51/ReadVariableOp_12?
>model_3/batch_normalization_53/FusedBatchNormV3/ReadVariableOp>model_3/batch_normalization_53/FusedBatchNormV3/ReadVariableOp2?
@model_3/batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1@model_3/batch_normalization_53/FusedBatchNormV3/ReadVariableOp_12^
-model_3/batch_normalization_53/ReadVariableOp-model_3/batch_normalization_53/ReadVariableOp2b
/model_3/batch_normalization_53/ReadVariableOp_1/model_3/batch_normalization_53/ReadVariableOp_12?
>model_3/batch_normalization_55/FusedBatchNormV3/ReadVariableOp>model_3/batch_normalization_55/FusedBatchNormV3/ReadVariableOp2?
@model_3/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1@model_3/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_12^
-model_3/batch_normalization_55/ReadVariableOp-model_3/batch_normalization_55/ReadVariableOp2b
/model_3/batch_normalization_55/ReadVariableOp_1/model_3/batch_normalization_55/ReadVariableOp_12T
(model_3/conv2d_46/BiasAdd/ReadVariableOp(model_3/conv2d_46/BiasAdd/ReadVariableOp2R
'model_3/conv2d_46/Conv2D/ReadVariableOp'model_3/conv2d_46/Conv2D/ReadVariableOp2T
(model_3/conv2d_48/BiasAdd/ReadVariableOp(model_3/conv2d_48/BiasAdd/ReadVariableOp2R
'model_3/conv2d_48/Conv2D/ReadVariableOp'model_3/conv2d_48/Conv2D/ReadVariableOp2T
(model_3/conv2d_50/BiasAdd/ReadVariableOp(model_3/conv2d_50/BiasAdd/ReadVariableOp2R
'model_3/conv2d_50/Conv2D/ReadVariableOp'model_3/conv2d_50/Conv2D/ReadVariableOp2T
(model_3/conv2d_52/BiasAdd/ReadVariableOp(model_3/conv2d_52/BiasAdd/ReadVariableOp2R
'model_3/conv2d_52/Conv2D/ReadVariableOp'model_3/conv2d_52/Conv2D/ReadVariableOp2T
(model_3/conv2d_54/BiasAdd/ReadVariableOp(model_3/conv2d_54/BiasAdd/ReadVariableOp2R
'model_3/conv2d_54/Conv2D/ReadVariableOp'model_3/conv2d_54/Conv2D/ReadVariableOp2T
(model_3/conv2d_56/BiasAdd/ReadVariableOp(model_3/conv2d_56/BiasAdd/ReadVariableOp2R
'model_3/conv2d_56/Conv2D/ReadVariableOp'model_3/conv2d_56/Conv2D/ReadVariableOp2T
(model_3/conv2d_58/BiasAdd/ReadVariableOp(model_3/conv2d_58/BiasAdd/ReadVariableOp2R
'model_3/conv2d_58/Conv2D/ReadVariableOp'model_3/conv2d_58/Conv2D/ReadVariableOp2T
(model_3/conv2d_59/BiasAdd/ReadVariableOp(model_3/conv2d_59/BiasAdd/ReadVariableOp2R
'model_3/conv2d_59/Conv2D/ReadVariableOp'model_3/conv2d_59/Conv2D/ReadVariableOp2h
2model_3/conv2d_transpose_10/BiasAdd/ReadVariableOp2model_3/conv2d_transpose_10/BiasAdd/ReadVariableOp2z
;model_3/conv2d_transpose_10/conv2d_transpose/ReadVariableOp;model_3/conv2d_transpose_10/conv2d_transpose/ReadVariableOp2h
2model_3/conv2d_transpose_11/BiasAdd/ReadVariableOp2model_3/conv2d_transpose_11/BiasAdd/ReadVariableOp2z
;model_3/conv2d_transpose_11/conv2d_transpose/ReadVariableOp;model_3/conv2d_transpose_11/conv2d_transpose/ReadVariableOp2f
1model_3/conv2d_transpose_9/BiasAdd/ReadVariableOp1model_3/conv2d_transpose_9/BiasAdd/ReadVariableOp2x
:model_3/conv2d_transpose_9/conv2d_transpose/ReadVariableOp:model_3/conv2d_transpose_9/conv2d_transpose/ReadVariableOp:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_4
?
~
)__inference_conv2d_48_layer_call_fn_80037

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
D__inference_conv2d_48_layer_call_and_return_conditional_losses_773892
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
?
?
Q__inference_batch_normalization_53_layer_call_and_return_conditional_losses_80891

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
d
H__inference_activation_51_layer_call_and_return_conditional_losses_80741

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
c
E__inference_dropout_19_layer_call_and_return_conditional_losses_80192

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
?
I
-__inference_activation_43_layer_call_fn_79989

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
H__inference_activation_43_layer_call_and_return_conditional_losses_773382
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
?
?
6__inference_batch_normalization_49_layer_call_fn_80537

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
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_767662
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
c
E__inference_dropout_18_layer_call_and_return_conditional_losses_80006

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
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_76885

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
?
?
D__inference_conv2d_52_layer_call_and_return_conditional_losses_77679

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
?
r
H__inference_concatenate_9_layer_call_and_return_conditional_losses_77793

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
??
?(
B__inference_model_3_layer_call_and_return_conditional_losses_79620

inputs,
(conv2d_46_conv2d_readvariableop_resource-
)conv2d_46_biasadd_readvariableop_resource2
.batch_normalization_43_readvariableop_resource4
0batch_normalization_43_readvariableop_1_resourceC
?batch_normalization_43_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_43_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_48_conv2d_readvariableop_resource-
)conv2d_48_biasadd_readvariableop_resource2
.batch_normalization_45_readvariableop_resource4
0batch_normalization_45_readvariableop_1_resourceC
?batch_normalization_45_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_45_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_50_conv2d_readvariableop_resource-
)conv2d_50_biasadd_readvariableop_resource2
.batch_normalization_47_readvariableop_resource4
0batch_normalization_47_readvariableop_1_resourceC
?batch_normalization_47_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_47_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_52_conv2d_readvariableop_resource-
)conv2d_52_biasadd_readvariableop_resource2
.batch_normalization_49_readvariableop_resource4
0batch_normalization_49_readvariableop_1_resourceC
?batch_normalization_49_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_49_fusedbatchnormv3_readvariableop_1_resource?
;conv2d_transpose_9_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_9_biasadd_readvariableop_resource,
(conv2d_54_conv2d_readvariableop_resource-
)conv2d_54_biasadd_readvariableop_resource2
.batch_normalization_51_readvariableop_resource4
0batch_normalization_51_readvariableop_1_resourceC
?batch_normalization_51_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_51_fusedbatchnormv3_readvariableop_1_resource@
<conv2d_transpose_10_conv2d_transpose_readvariableop_resource7
3conv2d_transpose_10_biasadd_readvariableop_resource,
(conv2d_56_conv2d_readvariableop_resource-
)conv2d_56_biasadd_readvariableop_resource2
.batch_normalization_53_readvariableop_resource4
0batch_normalization_53_readvariableop_1_resourceC
?batch_normalization_53_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_53_fusedbatchnormv3_readvariableop_1_resource@
<conv2d_transpose_11_conv2d_transpose_readvariableop_resource7
3conv2d_transpose_11_biasadd_readvariableop_resource,
(conv2d_58_conv2d_readvariableop_resource-
)conv2d_58_biasadd_readvariableop_resource2
.batch_normalization_55_readvariableop_resource4
0batch_normalization_55_readvariableop_1_resourceC
?batch_normalization_55_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_55_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_59_conv2d_readvariableop_resource-
)conv2d_59_biasadd_readvariableop_resource
identity??6batch_normalization_43/FusedBatchNormV3/ReadVariableOp?8batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_43/ReadVariableOp?'batch_normalization_43/ReadVariableOp_1?6batch_normalization_45/FusedBatchNormV3/ReadVariableOp?8batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_45/ReadVariableOp?'batch_normalization_45/ReadVariableOp_1?6batch_normalization_47/FusedBatchNormV3/ReadVariableOp?8batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_47/ReadVariableOp?'batch_normalization_47/ReadVariableOp_1?6batch_normalization_49/FusedBatchNormV3/ReadVariableOp?8batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_49/ReadVariableOp?'batch_normalization_49/ReadVariableOp_1?6batch_normalization_51/FusedBatchNormV3/ReadVariableOp?8batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_51/ReadVariableOp?'batch_normalization_51/ReadVariableOp_1?6batch_normalization_53/FusedBatchNormV3/ReadVariableOp?8batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_53/ReadVariableOp?'batch_normalization_53/ReadVariableOp_1?6batch_normalization_55/FusedBatchNormV3/ReadVariableOp?8batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_55/ReadVariableOp?'batch_normalization_55/ReadVariableOp_1? conv2d_46/BiasAdd/ReadVariableOp?conv2d_46/Conv2D/ReadVariableOp? conv2d_48/BiasAdd/ReadVariableOp?conv2d_48/Conv2D/ReadVariableOp? conv2d_50/BiasAdd/ReadVariableOp?conv2d_50/Conv2D/ReadVariableOp? conv2d_52/BiasAdd/ReadVariableOp?conv2d_52/Conv2D/ReadVariableOp? conv2d_54/BiasAdd/ReadVariableOp?conv2d_54/Conv2D/ReadVariableOp? conv2d_56/BiasAdd/ReadVariableOp?conv2d_56/Conv2D/ReadVariableOp? conv2d_58/BiasAdd/ReadVariableOp?conv2d_58/Conv2D/ReadVariableOp? conv2d_59/BiasAdd/ReadVariableOp?conv2d_59/Conv2D/ReadVariableOp?*conv2d_transpose_10/BiasAdd/ReadVariableOp?3conv2d_transpose_10/conv2d_transpose/ReadVariableOp?*conv2d_transpose_11/BiasAdd/ReadVariableOp?3conv2d_transpose_11/conv2d_transpose/ReadVariableOp?)conv2d_transpose_9/BiasAdd/ReadVariableOp?2conv2d_transpose_9/conv2d_transpose/ReadVariableOpg
CastCastinputs*

DstT0*

SrcT0*1
_output_shapes
:???????????2
Cast?
conv2d_46/Conv2D/ReadVariableOpReadVariableOp(conv2d_46_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_46/Conv2D/ReadVariableOp?
conv2d_46/Conv2D/CastCast'conv2d_46/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: 2
conv2d_46/Conv2D/Cast?
conv2d_46/Conv2DConv2DCast:y:0conv2d_46/Conv2D/Cast:y:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
conv2d_46/Conv2D?
 conv2d_46/BiasAdd/ReadVariableOpReadVariableOp)conv2d_46_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_46/BiasAdd/ReadVariableOp?
conv2d_46/BiasAdd/CastCast(conv2d_46/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: 2
conv2d_46/BiasAdd/Cast?
conv2d_46/BiasAddBiasAddconv2d_46/Conv2D:output:0conv2d_46/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:??????????? 2
conv2d_46/BiasAdd?
%batch_normalization_43/ReadVariableOpReadVariableOp.batch_normalization_43_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_43/ReadVariableOp?
'batch_normalization_43/ReadVariableOp_1ReadVariableOp0batch_normalization_43_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_43/ReadVariableOp_1?
6batch_normalization_43/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_43_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_43/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_43_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_43/FusedBatchNormV3FusedBatchNormV3conv2d_46/BiasAdd:output:0-batch_normalization_43/ReadVariableOp:value:0/batch_normalization_43/ReadVariableOp_1:value:0>batch_normalization_43/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2)
'batch_normalization_43/FusedBatchNormV3?
activation_43/ReluRelu+batch_normalization_43/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? 2
activation_43/Relu?
max_pooling2d_9/MaxPoolMaxPool activation_43/Relu:activations:0*
T0*1
_output_shapes
:??????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_9/MaxPool?
dropout_18/IdentityIdentity max_pooling2d_9/MaxPool:output:0*
T0*1
_output_shapes
:??????????? 2
dropout_18/Identity?
conv2d_48/Conv2D/ReadVariableOpReadVariableOp(conv2d_48_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_48/Conv2D/ReadVariableOp?
conv2d_48/Conv2D/CastCast'conv2d_48/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: @2
conv2d_48/Conv2D/Cast?
conv2d_48/Conv2DConv2Ddropout_18/Identity:output:0conv2d_48/Conv2D/Cast:y:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv2d_48/Conv2D?
 conv2d_48/BiasAdd/ReadVariableOpReadVariableOp)conv2d_48_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_48/BiasAdd/ReadVariableOp?
conv2d_48/BiasAdd/CastCast(conv2d_48/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@2
conv2d_48/BiasAdd/Cast?
conv2d_48/BiasAddBiasAddconv2d_48/Conv2D:output:0conv2d_48/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:???????????@2
conv2d_48/BiasAdd?
%batch_normalization_45/ReadVariableOpReadVariableOp.batch_normalization_45_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_45/ReadVariableOp?
'batch_normalization_45/ReadVariableOp_1ReadVariableOp0batch_normalization_45_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_45/ReadVariableOp_1?
6batch_normalization_45/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_45_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_45/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_45_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_45/FusedBatchNormV3FusedBatchNormV3conv2d_48/BiasAdd:output:0-batch_normalization_45/ReadVariableOp:value:0/batch_normalization_45/ReadVariableOp_1:value:0>batch_normalization_45/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2)
'batch_normalization_45/FusedBatchNormV3?
activation_45/ReluRelu+batch_normalization_45/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????@2
activation_45/Relu?
max_pooling2d_10/MaxPoolMaxPool activation_45/Relu:activations:0*
T0*1
_output_shapes
:???????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_10/MaxPool?
dropout_19/IdentityIdentity!max_pooling2d_10/MaxPool:output:0*
T0*1
_output_shapes
:???????????@2
dropout_19/Identity?
conv2d_50/Conv2D/ReadVariableOpReadVariableOp(conv2d_50_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02!
conv2d_50/Conv2D/ReadVariableOp?
conv2d_50/Conv2D/CastCast'conv2d_50/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*'
_output_shapes
:@?2
conv2d_50/Conv2D/Cast?
conv2d_50/Conv2DConv2Ddropout_19/Identity:output:0conv2d_50/Conv2D/Cast:y:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
conv2d_50/Conv2D?
 conv2d_50/BiasAdd/ReadVariableOpReadVariableOp)conv2d_50_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_50/BiasAdd/ReadVariableOp?
conv2d_50/BiasAdd/CastCast(conv2d_50/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:?2
conv2d_50/BiasAdd/Cast?
conv2d_50/BiasAddBiasAddconv2d_50/Conv2D:output:0conv2d_50/BiasAdd/Cast:y:0*
T0*2
_output_shapes 
:????????????2
conv2d_50/BiasAdd?
%batch_normalization_47/ReadVariableOpReadVariableOp.batch_normalization_47_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_47/ReadVariableOp?
'batch_normalization_47/ReadVariableOp_1ReadVariableOp0batch_normalization_47_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_47/ReadVariableOp_1?
6batch_normalization_47/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_47_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_47/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_47_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_47/FusedBatchNormV3FusedBatchNormV3conv2d_50/BiasAdd:output:0-batch_normalization_47/ReadVariableOp:value:0/batch_normalization_47/ReadVariableOp_1:value:0>batch_normalization_47/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2)
'batch_normalization_47/FusedBatchNormV3?
activation_47/ReluRelu+batch_normalization_47/FusedBatchNormV3:y:0*
T0*2
_output_shapes 
:????????????2
activation_47/Relu?
max_pooling2d_11/MaxPoolMaxPool activation_47/Relu:activations:0*
T0*0
_output_shapes
:?????????D\?*
ksize
*
paddingVALID*
strides
2
max_pooling2d_11/MaxPool?
dropout_20/IdentityIdentity!max_pooling2d_11/MaxPool:output:0*
T0*0
_output_shapes
:?????????D\?2
dropout_20/Identity?
conv2d_52/Conv2D/ReadVariableOpReadVariableOp(conv2d_52_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_52/Conv2D/ReadVariableOp?
conv2d_52/Conv2D/CastCast'conv2d_52/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*(
_output_shapes
:??2
conv2d_52/Conv2D/Cast?
conv2d_52/Conv2DConv2Ddropout_20/Identity:output:0conv2d_52/Conv2D/Cast:y:0*
T0*0
_output_shapes
:?????????D\?*
paddingSAME*
strides
2
conv2d_52/Conv2D?
 conv2d_52/BiasAdd/ReadVariableOpReadVariableOp)conv2d_52_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_52/BiasAdd/ReadVariableOp?
conv2d_52/BiasAdd/CastCast(conv2d_52/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:?2
conv2d_52/BiasAdd/Cast?
conv2d_52/BiasAddBiasAddconv2d_52/Conv2D:output:0conv2d_52/BiasAdd/Cast:y:0*
T0*0
_output_shapes
:?????????D\?2
conv2d_52/BiasAdd?
%batch_normalization_49/ReadVariableOpReadVariableOp.batch_normalization_49_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_49/ReadVariableOp?
'batch_normalization_49/ReadVariableOp_1ReadVariableOp0batch_normalization_49_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_49/ReadVariableOp_1?
6batch_normalization_49/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_49_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_49/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_49_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_49/FusedBatchNormV3FusedBatchNormV3conv2d_52/BiasAdd:output:0-batch_normalization_49/ReadVariableOp:value:0/batch_normalization_49/ReadVariableOp_1:value:0>batch_normalization_49/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????D\?:?:?:?:?:*
epsilon%o?:*
is_training( 2)
'batch_normalization_49/FusedBatchNormV3?
activation_49/ReluRelu+batch_normalization_49/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????D\?2
activation_49/Relu?
conv2d_transpose_9/ShapeShape activation_49/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_9/Shape?
&conv2d_transpose_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_9/strided_slice/stack?
(conv2d_transpose_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_9/strided_slice/stack_1?
(conv2d_transpose_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_9/strided_slice/stack_2?
 conv2d_transpose_9/strided_sliceStridedSlice!conv2d_transpose_9/Shape:output:0/conv2d_transpose_9/strided_slice/stack:output:01conv2d_transpose_9/strided_slice/stack_1:output:01conv2d_transpose_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_9/strided_slice{
conv2d_transpose_9/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_9/stack/1{
conv2d_transpose_9/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_9/stack/2{
conv2d_transpose_9/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_9/stack/3?
conv2d_transpose_9/stackPack)conv2d_transpose_9/strided_slice:output:0#conv2d_transpose_9/stack/1:output:0#conv2d_transpose_9/stack/2:output:0#conv2d_transpose_9/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_9/stack?
(conv2d_transpose_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_9/strided_slice_1/stack?
*conv2d_transpose_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_9/strided_slice_1/stack_1?
*conv2d_transpose_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_9/strided_slice_1/stack_2?
"conv2d_transpose_9/strided_slice_1StridedSlice!conv2d_transpose_9/stack:output:01conv2d_transpose_9/strided_slice_1/stack:output:03conv2d_transpose_9/strided_slice_1/stack_1:output:03conv2d_transpose_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_9/strided_slice_1?
2conv2d_transpose_9/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_9_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype024
2conv2d_transpose_9/conv2d_transpose/ReadVariableOp?
(conv2d_transpose_9/conv2d_transpose/CastCast:conv2d_transpose_9/conv2d_transpose/ReadVariableOp:value:0*

DstT0*

SrcT0*(
_output_shapes
:??2*
(conv2d_transpose_9/conv2d_transpose/Cast?
#conv2d_transpose_9/conv2d_transposeConv2DBackpropInput!conv2d_transpose_9/stack:output:0,conv2d_transpose_9/conv2d_transpose/Cast:y:0 activation_49/Relu:activations:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2%
#conv2d_transpose_9/conv2d_transpose?
)conv2d_transpose_9/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_9_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)conv2d_transpose_9/BiasAdd/ReadVariableOp?
conv2d_transpose_9/BiasAdd/CastCast1conv2d_transpose_9/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:?2!
conv2d_transpose_9/BiasAdd/Cast?
conv2d_transpose_9/BiasAddBiasAdd,conv2d_transpose_9/conv2d_transpose:output:0#conv2d_transpose_9/BiasAdd/Cast:y:0*
T0*2
_output_shapes 
:????????????2
conv2d_transpose_9/BiasAddx
concatenate_9/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_9/concat/axis?
concatenate_9/concatConcatV2#conv2d_transpose_9/BiasAdd:output:0 activation_47/Relu:activations:0"concatenate_9/concat/axis:output:0*
N*
T0*2
_output_shapes 
:????????????2
concatenate_9/concat?
dropout_21/IdentityIdentityconcatenate_9/concat:output:0*
T0*2
_output_shapes 
:????????????2
dropout_21/Identity?
conv2d_54/Conv2D/ReadVariableOpReadVariableOp(conv2d_54_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_54/Conv2D/ReadVariableOp?
conv2d_54/Conv2D/CastCast'conv2d_54/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*(
_output_shapes
:??2
conv2d_54/Conv2D/Cast?
conv2d_54/Conv2DConv2Ddropout_21/Identity:output:0conv2d_54/Conv2D/Cast:y:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
conv2d_54/Conv2D?
 conv2d_54/BiasAdd/ReadVariableOpReadVariableOp)conv2d_54_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_54/BiasAdd/ReadVariableOp?
conv2d_54/BiasAdd/CastCast(conv2d_54/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:?2
conv2d_54/BiasAdd/Cast?
conv2d_54/BiasAddBiasAddconv2d_54/Conv2D:output:0conv2d_54/BiasAdd/Cast:y:0*
T0*2
_output_shapes 
:????????????2
conv2d_54/BiasAdd?
%batch_normalization_51/ReadVariableOpReadVariableOp.batch_normalization_51_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_51/ReadVariableOp?
'batch_normalization_51/ReadVariableOp_1ReadVariableOp0batch_normalization_51_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_51/ReadVariableOp_1?
6batch_normalization_51/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_51_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_51/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_51_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_51/FusedBatchNormV3FusedBatchNormV3conv2d_54/BiasAdd:output:0-batch_normalization_51/ReadVariableOp:value:0/batch_normalization_51/ReadVariableOp_1:value:0>batch_normalization_51/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2)
'batch_normalization_51/FusedBatchNormV3?
activation_51/ReluRelu+batch_normalization_51/FusedBatchNormV3:y:0*
T0*2
_output_shapes 
:????????????2
activation_51/Relu?
conv2d_transpose_10/ShapeShape activation_51/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_10/Shape?
'conv2d_transpose_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_10/strided_slice/stack?
)conv2d_transpose_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_10/strided_slice/stack_1?
)conv2d_transpose_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_10/strided_slice/stack_2?
!conv2d_transpose_10/strided_sliceStridedSlice"conv2d_transpose_10/Shape:output:00conv2d_transpose_10/strided_slice/stack:output:02conv2d_transpose_10/strided_slice/stack_1:output:02conv2d_transpose_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_10/strided_slice}
conv2d_transpose_10/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_10/stack/1}
conv2d_transpose_10/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_10/stack/2|
conv2d_transpose_10/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_10/stack/3?
conv2d_transpose_10/stackPack*conv2d_transpose_10/strided_slice:output:0$conv2d_transpose_10/stack/1:output:0$conv2d_transpose_10/stack/2:output:0$conv2d_transpose_10/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_10/stack?
)conv2d_transpose_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_10/strided_slice_1/stack?
+conv2d_transpose_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_10/strided_slice_1/stack_1?
+conv2d_transpose_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_10/strided_slice_1/stack_2?
#conv2d_transpose_10/strided_slice_1StridedSlice"conv2d_transpose_10/stack:output:02conv2d_transpose_10/strided_slice_1/stack:output:04conv2d_transpose_10/strided_slice_1/stack_1:output:04conv2d_transpose_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_10/strided_slice_1?
3conv2d_transpose_10/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_10_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype025
3conv2d_transpose_10/conv2d_transpose/ReadVariableOp?
)conv2d_transpose_10/conv2d_transpose/CastCast;conv2d_transpose_10/conv2d_transpose/ReadVariableOp:value:0*

DstT0*

SrcT0*'
_output_shapes
:@?2+
)conv2d_transpose_10/conv2d_transpose/Cast?
$conv2d_transpose_10/conv2d_transposeConv2DBackpropInput"conv2d_transpose_10/stack:output:0-conv2d_transpose_10/conv2d_transpose/Cast:y:0 activation_51/Relu:activations:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2&
$conv2d_transpose_10/conv2d_transpose?
*conv2d_transpose_10/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*conv2d_transpose_10/BiasAdd/ReadVariableOp?
 conv2d_transpose_10/BiasAdd/CastCast2conv2d_transpose_10/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@2"
 conv2d_transpose_10/BiasAdd/Cast?
conv2d_transpose_10/BiasAddBiasAdd-conv2d_transpose_10/conv2d_transpose:output:0$conv2d_transpose_10/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:???????????@2
conv2d_transpose_10/BiasAddz
concatenate_10/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_10/concat/axis?
concatenate_10/concatConcatV2$conv2d_transpose_10/BiasAdd:output:0 activation_45/Relu:activations:0#concatenate_10/concat/axis:output:0*
N*
T0*2
_output_shapes 
:????????????2
concatenate_10/concat?
dropout_22/IdentityIdentityconcatenate_10/concat:output:0*
T0*2
_output_shapes 
:????????????2
dropout_22/Identity?
conv2d_56/Conv2D/ReadVariableOpReadVariableOp(conv2d_56_conv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02!
conv2d_56/Conv2D/ReadVariableOp?
conv2d_56/Conv2D/CastCast'conv2d_56/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*'
_output_shapes
:?@2
conv2d_56/Conv2D/Cast?
conv2d_56/Conv2DConv2Ddropout_22/Identity:output:0conv2d_56/Conv2D/Cast:y:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv2d_56/Conv2D?
 conv2d_56/BiasAdd/ReadVariableOpReadVariableOp)conv2d_56_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_56/BiasAdd/ReadVariableOp?
conv2d_56/BiasAdd/CastCast(conv2d_56/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@2
conv2d_56/BiasAdd/Cast?
conv2d_56/BiasAddBiasAddconv2d_56/Conv2D:output:0conv2d_56/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:???????????@2
conv2d_56/BiasAdd?
%batch_normalization_53/ReadVariableOpReadVariableOp.batch_normalization_53_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_53/ReadVariableOp?
'batch_normalization_53/ReadVariableOp_1ReadVariableOp0batch_normalization_53_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_53/ReadVariableOp_1?
6batch_normalization_53/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_53_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_53/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_53_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_53/FusedBatchNormV3FusedBatchNormV3conv2d_56/BiasAdd:output:0-batch_normalization_53/ReadVariableOp:value:0/batch_normalization_53/ReadVariableOp_1:value:0>batch_normalization_53/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2)
'batch_normalization_53/FusedBatchNormV3?
activation_53/ReluRelu+batch_normalization_53/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????@2
activation_53/Relu?
conv2d_transpose_11/ShapeShape activation_53/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_11/Shape?
'conv2d_transpose_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_11/strided_slice/stack?
)conv2d_transpose_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_11/strided_slice/stack_1?
)conv2d_transpose_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_11/strided_slice/stack_2?
!conv2d_transpose_11/strided_sliceStridedSlice"conv2d_transpose_11/Shape:output:00conv2d_transpose_11/strided_slice/stack:output:02conv2d_transpose_11/strided_slice/stack_1:output:02conv2d_transpose_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_11/strided_slice}
conv2d_transpose_11/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_11/stack/1}
conv2d_transpose_11/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_11/stack/2|
conv2d_transpose_11/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_11/stack/3?
conv2d_transpose_11/stackPack*conv2d_transpose_11/strided_slice:output:0$conv2d_transpose_11/stack/1:output:0$conv2d_transpose_11/stack/2:output:0$conv2d_transpose_11/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_11/stack?
)conv2d_transpose_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_11/strided_slice_1/stack?
+conv2d_transpose_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_11/strided_slice_1/stack_1?
+conv2d_transpose_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_11/strided_slice_1/stack_2?
#conv2d_transpose_11/strided_slice_1StridedSlice"conv2d_transpose_11/stack:output:02conv2d_transpose_11/strided_slice_1/stack:output:04conv2d_transpose_11/strided_slice_1/stack_1:output:04conv2d_transpose_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_11/strided_slice_1?
3conv2d_transpose_11/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_11_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype025
3conv2d_transpose_11/conv2d_transpose/ReadVariableOp?
)conv2d_transpose_11/conv2d_transpose/CastCast;conv2d_transpose_11/conv2d_transpose/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: @2+
)conv2d_transpose_11/conv2d_transpose/Cast?
$conv2d_transpose_11/conv2d_transposeConv2DBackpropInput"conv2d_transpose_11/stack:output:0-conv2d_transpose_11/conv2d_transpose/Cast:y:0 activation_53/Relu:activations:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2&
$conv2d_transpose_11/conv2d_transpose?
*conv2d_transpose_11/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*conv2d_transpose_11/BiasAdd/ReadVariableOp?
 conv2d_transpose_11/BiasAdd/CastCast2conv2d_transpose_11/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 conv2d_transpose_11/BiasAdd/Cast?
conv2d_transpose_11/BiasAddBiasAdd-conv2d_transpose_11/conv2d_transpose:output:0$conv2d_transpose_11/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:??????????? 2
conv2d_transpose_11/BiasAddz
concatenate_11/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_11/concat/axis?
concatenate_11/concatConcatV2$conv2d_transpose_11/BiasAdd:output:0 activation_43/Relu:activations:0#concatenate_11/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????@2
concatenate_11/concat?
dropout_23/IdentityIdentityconcatenate_11/concat:output:0*
T0*1
_output_shapes
:???????????@2
dropout_23/Identity?
conv2d_58/Conv2D/ReadVariableOpReadVariableOp(conv2d_58_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02!
conv2d_58/Conv2D/ReadVariableOp?
conv2d_58/Conv2D/CastCast'conv2d_58/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
:@ 2
conv2d_58/Conv2D/Cast?
conv2d_58/Conv2DConv2Ddropout_23/Identity:output:0conv2d_58/Conv2D/Cast:y:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
conv2d_58/Conv2D?
 conv2d_58/BiasAdd/ReadVariableOpReadVariableOp)conv2d_58_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_58/BiasAdd/ReadVariableOp?
conv2d_58/BiasAdd/CastCast(conv2d_58/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: 2
conv2d_58/BiasAdd/Cast?
conv2d_58/BiasAddBiasAddconv2d_58/Conv2D:output:0conv2d_58/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:??????????? 2
conv2d_58/BiasAdd?
%batch_normalization_55/ReadVariableOpReadVariableOp.batch_normalization_55_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_55/ReadVariableOp?
'batch_normalization_55/ReadVariableOp_1ReadVariableOp0batch_normalization_55_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_55/ReadVariableOp_1?
6batch_normalization_55/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_55_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_55/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_55_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_55/FusedBatchNormV3FusedBatchNormV3conv2d_58/BiasAdd:output:0-batch_normalization_55/ReadVariableOp:value:0/batch_normalization_55/ReadVariableOp_1:value:0>batch_normalization_55/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2)
'batch_normalization_55/FusedBatchNormV3?
activation_55/ReluRelu+batch_normalization_55/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? 2
activation_55/Relu?
conv2d_59/Conv2D/ReadVariableOpReadVariableOp(conv2d_59_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_59/Conv2D/ReadVariableOp?
conv2d_59/Conv2D/CastCast'conv2d_59/Conv2D/ReadVariableOp:value:0*

DstT0*

SrcT0*&
_output_shapes
: 2
conv2d_59/Conv2D/Cast?
conv2d_59/Conv2DConv2D activation_55/Relu:activations:0conv2d_59/Conv2D/Cast:y:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
conv2d_59/Conv2D?
 conv2d_59/BiasAdd/ReadVariableOpReadVariableOp)conv2d_59_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_59/BiasAdd/ReadVariableOp?
conv2d_59/BiasAdd/CastCast(conv2d_59/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:2
conv2d_59/BiasAdd/Cast?
conv2d_59/BiasAddBiasAddconv2d_59/Conv2D:output:0conv2d_59/BiasAdd/Cast:y:0*
T0*1
_output_shapes
:???????????2
conv2d_59/BiasAdd?
conv2d_59/SigmoidSigmoidconv2d_59/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_59/Sigmoid?
IdentityIdentityconv2d_59/Sigmoid:y:07^batch_normalization_43/FusedBatchNormV3/ReadVariableOp9^batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_43/ReadVariableOp(^batch_normalization_43/ReadVariableOp_17^batch_normalization_45/FusedBatchNormV3/ReadVariableOp9^batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_45/ReadVariableOp(^batch_normalization_45/ReadVariableOp_17^batch_normalization_47/FusedBatchNormV3/ReadVariableOp9^batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_47/ReadVariableOp(^batch_normalization_47/ReadVariableOp_17^batch_normalization_49/FusedBatchNormV3/ReadVariableOp9^batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_49/ReadVariableOp(^batch_normalization_49/ReadVariableOp_17^batch_normalization_51/FusedBatchNormV3/ReadVariableOp9^batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_51/ReadVariableOp(^batch_normalization_51/ReadVariableOp_17^batch_normalization_53/FusedBatchNormV3/ReadVariableOp9^batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_53/ReadVariableOp(^batch_normalization_53/ReadVariableOp_17^batch_normalization_55/FusedBatchNormV3/ReadVariableOp9^batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_55/ReadVariableOp(^batch_normalization_55/ReadVariableOp_1!^conv2d_46/BiasAdd/ReadVariableOp ^conv2d_46/Conv2D/ReadVariableOp!^conv2d_48/BiasAdd/ReadVariableOp ^conv2d_48/Conv2D/ReadVariableOp!^conv2d_50/BiasAdd/ReadVariableOp ^conv2d_50/Conv2D/ReadVariableOp!^conv2d_52/BiasAdd/ReadVariableOp ^conv2d_52/Conv2D/ReadVariableOp!^conv2d_54/BiasAdd/ReadVariableOp ^conv2d_54/Conv2D/ReadVariableOp!^conv2d_56/BiasAdd/ReadVariableOp ^conv2d_56/Conv2D/ReadVariableOp!^conv2d_58/BiasAdd/ReadVariableOp ^conv2d_58/Conv2D/ReadVariableOp!^conv2d_59/BiasAdd/ReadVariableOp ^conv2d_59/Conv2D/ReadVariableOp+^conv2d_transpose_10/BiasAdd/ReadVariableOp4^conv2d_transpose_10/conv2d_transpose/ReadVariableOp+^conv2d_transpose_11/BiasAdd/ReadVariableOp4^conv2d_transpose_11/conv2d_transpose/ReadVariableOp*^conv2d_transpose_9/BiasAdd/ReadVariableOp3^conv2d_transpose_9/conv2d_transpose/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::::::::::2p
6batch_normalization_43/FusedBatchNormV3/ReadVariableOp6batch_normalization_43/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_43/FusedBatchNormV3/ReadVariableOp_18batch_normalization_43/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_43/ReadVariableOp%batch_normalization_43/ReadVariableOp2R
'batch_normalization_43/ReadVariableOp_1'batch_normalization_43/ReadVariableOp_12p
6batch_normalization_45/FusedBatchNormV3/ReadVariableOp6batch_normalization_45/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_45/FusedBatchNormV3/ReadVariableOp_18batch_normalization_45/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_45/ReadVariableOp%batch_normalization_45/ReadVariableOp2R
'batch_normalization_45/ReadVariableOp_1'batch_normalization_45/ReadVariableOp_12p
6batch_normalization_47/FusedBatchNormV3/ReadVariableOp6batch_normalization_47/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_47/FusedBatchNormV3/ReadVariableOp_18batch_normalization_47/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_47/ReadVariableOp%batch_normalization_47/ReadVariableOp2R
'batch_normalization_47/ReadVariableOp_1'batch_normalization_47/ReadVariableOp_12p
6batch_normalization_49/FusedBatchNormV3/ReadVariableOp6batch_normalization_49/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_49/FusedBatchNormV3/ReadVariableOp_18batch_normalization_49/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_49/ReadVariableOp%batch_normalization_49/ReadVariableOp2R
'batch_normalization_49/ReadVariableOp_1'batch_normalization_49/ReadVariableOp_12p
6batch_normalization_51/FusedBatchNormV3/ReadVariableOp6batch_normalization_51/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_51/FusedBatchNormV3/ReadVariableOp_18batch_normalization_51/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_51/ReadVariableOp%batch_normalization_51/ReadVariableOp2R
'batch_normalization_51/ReadVariableOp_1'batch_normalization_51/ReadVariableOp_12p
6batch_normalization_53/FusedBatchNormV3/ReadVariableOp6batch_normalization_53/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_53/FusedBatchNormV3/ReadVariableOp_18batch_normalization_53/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_53/ReadVariableOp%batch_normalization_53/ReadVariableOp2R
'batch_normalization_53/ReadVariableOp_1'batch_normalization_53/ReadVariableOp_12p
6batch_normalization_55/FusedBatchNormV3/ReadVariableOp6batch_normalization_55/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_55/FusedBatchNormV3/ReadVariableOp_18batch_normalization_55/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_55/ReadVariableOp%batch_normalization_55/ReadVariableOp2R
'batch_normalization_55/ReadVariableOp_1'batch_normalization_55/ReadVariableOp_12D
 conv2d_46/BiasAdd/ReadVariableOp conv2d_46/BiasAdd/ReadVariableOp2B
conv2d_46/Conv2D/ReadVariableOpconv2d_46/Conv2D/ReadVariableOp2D
 conv2d_48/BiasAdd/ReadVariableOp conv2d_48/BiasAdd/ReadVariableOp2B
conv2d_48/Conv2D/ReadVariableOpconv2d_48/Conv2D/ReadVariableOp2D
 conv2d_50/BiasAdd/ReadVariableOp conv2d_50/BiasAdd/ReadVariableOp2B
conv2d_50/Conv2D/ReadVariableOpconv2d_50/Conv2D/ReadVariableOp2D
 conv2d_52/BiasAdd/ReadVariableOp conv2d_52/BiasAdd/ReadVariableOp2B
conv2d_52/Conv2D/ReadVariableOpconv2d_52/Conv2D/ReadVariableOp2D
 conv2d_54/BiasAdd/ReadVariableOp conv2d_54/BiasAdd/ReadVariableOp2B
conv2d_54/Conv2D/ReadVariableOpconv2d_54/Conv2D/ReadVariableOp2D
 conv2d_56/BiasAdd/ReadVariableOp conv2d_56/BiasAdd/ReadVariableOp2B
conv2d_56/Conv2D/ReadVariableOpconv2d_56/Conv2D/ReadVariableOp2D
 conv2d_58/BiasAdd/ReadVariableOp conv2d_58/BiasAdd/ReadVariableOp2B
conv2d_58/Conv2D/ReadVariableOpconv2d_58/Conv2D/ReadVariableOp2D
 conv2d_59/BiasAdd/ReadVariableOp conv2d_59/BiasAdd/ReadVariableOp2B
conv2d_59/Conv2D/ReadVariableOpconv2d_59/Conv2D/ReadVariableOp2X
*conv2d_transpose_10/BiasAdd/ReadVariableOp*conv2d_transpose_10/BiasAdd/ReadVariableOp2j
3conv2d_transpose_10/conv2d_transpose/ReadVariableOp3conv2d_transpose_10/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_11/BiasAdd/ReadVariableOp*conv2d_transpose_11/BiasAdd/ReadVariableOp2j
3conv2d_transpose_11/conv2d_transpose/ReadVariableOp3conv2d_transpose_11/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_9/BiasAdd/ReadVariableOp)conv2d_transpose_9/BiasAdd/ReadVariableOp2h
2conv2d_transpose_9/conv2d_transpose/ReadVariableOp2conv2d_transpose_9/conv2d_transpose/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_77732

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
Q__inference_batch_normalization_55_layer_call_and_return_conditional_losses_78227

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
?
?
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_80121

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
)__inference_conv2d_58_layer_call_fn_81006

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
D__inference_conv2d_58_layer_call_and_return_conditional_losses_781742
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
?
d
E__inference_dropout_18_layer_call_and_return_conditional_losses_80001

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
?
?
Q__inference_batch_normalization_53_layer_call_and_return_conditional_losses_78062

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
?
c
E__inference_dropout_23_layer_call_and_return_conditional_losses_78149

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
?
?
D__inference_conv2d_48_layer_call_and_return_conditional_losses_77389

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
?
?
6__inference_batch_normalization_45_layer_call_fn_80165

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
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_765342
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
?
F
*__inference_dropout_22_layer_call_fn_80786

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
E__inference_dropout_22_layer_call_and_return_conditional_losses_779842
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
?
?
6__inference_batch_normalization_53_layer_call_fn_80871

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
Q__inference_batch_normalization_53_layer_call_and_return_conditional_losses_770662
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
?
?
6__inference_batch_normalization_47_layer_call_fn_80274

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
Q__inference_batch_normalization_47_layer_call_and_return_conditional_losses_775692
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
?
?
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_77897

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
?
6__inference_batch_normalization_45_layer_call_fn_80101

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
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_774422
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
??
?B
__inference__traced_save_81588
file_prefix/
+savev2_conv2d_46_kernel_read_readvariableop-
)savev2_conv2d_46_bias_read_readvariableop;
7savev2_batch_normalization_43_gamma_read_readvariableop:
6savev2_batch_normalization_43_beta_read_readvariableopA
=savev2_batch_normalization_43_moving_mean_read_readvariableopE
Asavev2_batch_normalization_43_moving_variance_read_readvariableop/
+savev2_conv2d_48_kernel_read_readvariableop-
)savev2_conv2d_48_bias_read_readvariableop;
7savev2_batch_normalization_45_gamma_read_readvariableop:
6savev2_batch_normalization_45_beta_read_readvariableopA
=savev2_batch_normalization_45_moving_mean_read_readvariableopE
Asavev2_batch_normalization_45_moving_variance_read_readvariableop/
+savev2_conv2d_50_kernel_read_readvariableop-
)savev2_conv2d_50_bias_read_readvariableop;
7savev2_batch_normalization_47_gamma_read_readvariableop:
6savev2_batch_normalization_47_beta_read_readvariableopA
=savev2_batch_normalization_47_moving_mean_read_readvariableopE
Asavev2_batch_normalization_47_moving_variance_read_readvariableop/
+savev2_conv2d_52_kernel_read_readvariableop-
)savev2_conv2d_52_bias_read_readvariableop;
7savev2_batch_normalization_49_gamma_read_readvariableop:
6savev2_batch_normalization_49_beta_read_readvariableopA
=savev2_batch_normalization_49_moving_mean_read_readvariableopE
Asavev2_batch_normalization_49_moving_variance_read_readvariableop8
4savev2_conv2d_transpose_9_kernel_read_readvariableop6
2savev2_conv2d_transpose_9_bias_read_readvariableop/
+savev2_conv2d_54_kernel_read_readvariableop-
)savev2_conv2d_54_bias_read_readvariableop;
7savev2_batch_normalization_51_gamma_read_readvariableop:
6savev2_batch_normalization_51_beta_read_readvariableopA
=savev2_batch_normalization_51_moving_mean_read_readvariableopE
Asavev2_batch_normalization_51_moving_variance_read_readvariableop9
5savev2_conv2d_transpose_10_kernel_read_readvariableop7
3savev2_conv2d_transpose_10_bias_read_readvariableop/
+savev2_conv2d_56_kernel_read_readvariableop-
)savev2_conv2d_56_bias_read_readvariableop;
7savev2_batch_normalization_53_gamma_read_readvariableop:
6savev2_batch_normalization_53_beta_read_readvariableopA
=savev2_batch_normalization_53_moving_mean_read_readvariableopE
Asavev2_batch_normalization_53_moving_variance_read_readvariableop9
5savev2_conv2d_transpose_11_kernel_read_readvariableop7
3savev2_conv2d_transpose_11_bias_read_readvariableop/
+savev2_conv2d_58_kernel_read_readvariableop-
)savev2_conv2d_58_bias_read_readvariableop;
7savev2_batch_normalization_55_gamma_read_readvariableop:
6savev2_batch_normalization_55_beta_read_readvariableopA
=savev2_batch_normalization_55_moving_mean_read_readvariableopE
Asavev2_batch_normalization_55_moving_variance_read_readvariableop/
+savev2_conv2d_59_kernel_read_readvariableop-
)savev2_conv2d_59_bias_read_readvariableop/
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
9savev2_cond_1_adam_conv2d_46_kernel_m_read_readvariableop;
7savev2_cond_1_adam_conv2d_46_bias_m_read_readvariableopI
Esavev2_cond_1_adam_batch_normalization_43_gamma_m_read_readvariableopH
Dsavev2_cond_1_adam_batch_normalization_43_beta_m_read_readvariableop=
9savev2_cond_1_adam_conv2d_48_kernel_m_read_readvariableop;
7savev2_cond_1_adam_conv2d_48_bias_m_read_readvariableopI
Esavev2_cond_1_adam_batch_normalization_45_gamma_m_read_readvariableopH
Dsavev2_cond_1_adam_batch_normalization_45_beta_m_read_readvariableop=
9savev2_cond_1_adam_conv2d_50_kernel_m_read_readvariableop;
7savev2_cond_1_adam_conv2d_50_bias_m_read_readvariableopI
Esavev2_cond_1_adam_batch_normalization_47_gamma_m_read_readvariableopH
Dsavev2_cond_1_adam_batch_normalization_47_beta_m_read_readvariableop=
9savev2_cond_1_adam_conv2d_52_kernel_m_read_readvariableop;
7savev2_cond_1_adam_conv2d_52_bias_m_read_readvariableopI
Esavev2_cond_1_adam_batch_normalization_49_gamma_m_read_readvariableopH
Dsavev2_cond_1_adam_batch_normalization_49_beta_m_read_readvariableopF
Bsavev2_cond_1_adam_conv2d_transpose_9_kernel_m_read_readvariableopD
@savev2_cond_1_adam_conv2d_transpose_9_bias_m_read_readvariableop=
9savev2_cond_1_adam_conv2d_54_kernel_m_read_readvariableop;
7savev2_cond_1_adam_conv2d_54_bias_m_read_readvariableopI
Esavev2_cond_1_adam_batch_normalization_51_gamma_m_read_readvariableopH
Dsavev2_cond_1_adam_batch_normalization_51_beta_m_read_readvariableopG
Csavev2_cond_1_adam_conv2d_transpose_10_kernel_m_read_readvariableopE
Asavev2_cond_1_adam_conv2d_transpose_10_bias_m_read_readvariableop=
9savev2_cond_1_adam_conv2d_56_kernel_m_read_readvariableop;
7savev2_cond_1_adam_conv2d_56_bias_m_read_readvariableopI
Esavev2_cond_1_adam_batch_normalization_53_gamma_m_read_readvariableopH
Dsavev2_cond_1_adam_batch_normalization_53_beta_m_read_readvariableopG
Csavev2_cond_1_adam_conv2d_transpose_11_kernel_m_read_readvariableopE
Asavev2_cond_1_adam_conv2d_transpose_11_bias_m_read_readvariableop=
9savev2_cond_1_adam_conv2d_58_kernel_m_read_readvariableop;
7savev2_cond_1_adam_conv2d_58_bias_m_read_readvariableopI
Esavev2_cond_1_adam_batch_normalization_55_gamma_m_read_readvariableopH
Dsavev2_cond_1_adam_batch_normalization_55_beta_m_read_readvariableop=
9savev2_cond_1_adam_conv2d_59_kernel_m_read_readvariableop;
7savev2_cond_1_adam_conv2d_59_bias_m_read_readvariableop=
9savev2_cond_1_adam_conv2d_46_kernel_v_read_readvariableop;
7savev2_cond_1_adam_conv2d_46_bias_v_read_readvariableopI
Esavev2_cond_1_adam_batch_normalization_43_gamma_v_read_readvariableopH
Dsavev2_cond_1_adam_batch_normalization_43_beta_v_read_readvariableop=
9savev2_cond_1_adam_conv2d_48_kernel_v_read_readvariableop;
7savev2_cond_1_adam_conv2d_48_bias_v_read_readvariableopI
Esavev2_cond_1_adam_batch_normalization_45_gamma_v_read_readvariableopH
Dsavev2_cond_1_adam_batch_normalization_45_beta_v_read_readvariableop=
9savev2_cond_1_adam_conv2d_50_kernel_v_read_readvariableop;
7savev2_cond_1_adam_conv2d_50_bias_v_read_readvariableopI
Esavev2_cond_1_adam_batch_normalization_47_gamma_v_read_readvariableopH
Dsavev2_cond_1_adam_batch_normalization_47_beta_v_read_readvariableop=
9savev2_cond_1_adam_conv2d_52_kernel_v_read_readvariableop;
7savev2_cond_1_adam_conv2d_52_bias_v_read_readvariableopI
Esavev2_cond_1_adam_batch_normalization_49_gamma_v_read_readvariableopH
Dsavev2_cond_1_adam_batch_normalization_49_beta_v_read_readvariableopF
Bsavev2_cond_1_adam_conv2d_transpose_9_kernel_v_read_readvariableopD
@savev2_cond_1_adam_conv2d_transpose_9_bias_v_read_readvariableop=
9savev2_cond_1_adam_conv2d_54_kernel_v_read_readvariableop;
7savev2_cond_1_adam_conv2d_54_bias_v_read_readvariableopI
Esavev2_cond_1_adam_batch_normalization_51_gamma_v_read_readvariableopH
Dsavev2_cond_1_adam_batch_normalization_51_beta_v_read_readvariableopG
Csavev2_cond_1_adam_conv2d_transpose_10_kernel_v_read_readvariableopE
Asavev2_cond_1_adam_conv2d_transpose_10_bias_v_read_readvariableop=
9savev2_cond_1_adam_conv2d_56_kernel_v_read_readvariableop;
7savev2_cond_1_adam_conv2d_56_bias_v_read_readvariableopI
Esavev2_cond_1_adam_batch_normalization_53_gamma_v_read_readvariableopH
Dsavev2_cond_1_adam_batch_normalization_53_beta_v_read_readvariableopG
Csavev2_cond_1_adam_conv2d_transpose_11_kernel_v_read_readvariableopE
Asavev2_cond_1_adam_conv2d_transpose_11_bias_v_read_readvariableop=
9savev2_cond_1_adam_conv2d_58_kernel_v_read_readvariableop;
7savev2_cond_1_adam_conv2d_58_bias_v_read_readvariableopI
Esavev2_cond_1_adam_batch_normalization_55_gamma_v_read_readvariableopH
Dsavev2_cond_1_adam_batch_normalization_55_beta_v_read_readvariableop=
9savev2_cond_1_adam_conv2d_59_kernel_v_read_readvariableop;
7savev2_cond_1_adam_conv2d_59_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_46_kernel_read_readvariableop)savev2_conv2d_46_bias_read_readvariableop7savev2_batch_normalization_43_gamma_read_readvariableop6savev2_batch_normalization_43_beta_read_readvariableop=savev2_batch_normalization_43_moving_mean_read_readvariableopAsavev2_batch_normalization_43_moving_variance_read_readvariableop+savev2_conv2d_48_kernel_read_readvariableop)savev2_conv2d_48_bias_read_readvariableop7savev2_batch_normalization_45_gamma_read_readvariableop6savev2_batch_normalization_45_beta_read_readvariableop=savev2_batch_normalization_45_moving_mean_read_readvariableopAsavev2_batch_normalization_45_moving_variance_read_readvariableop+savev2_conv2d_50_kernel_read_readvariableop)savev2_conv2d_50_bias_read_readvariableop7savev2_batch_normalization_47_gamma_read_readvariableop6savev2_batch_normalization_47_beta_read_readvariableop=savev2_batch_normalization_47_moving_mean_read_readvariableopAsavev2_batch_normalization_47_moving_variance_read_readvariableop+savev2_conv2d_52_kernel_read_readvariableop)savev2_conv2d_52_bias_read_readvariableop7savev2_batch_normalization_49_gamma_read_readvariableop6savev2_batch_normalization_49_beta_read_readvariableop=savev2_batch_normalization_49_moving_mean_read_readvariableopAsavev2_batch_normalization_49_moving_variance_read_readvariableop4savev2_conv2d_transpose_9_kernel_read_readvariableop2savev2_conv2d_transpose_9_bias_read_readvariableop+savev2_conv2d_54_kernel_read_readvariableop)savev2_conv2d_54_bias_read_readvariableop7savev2_batch_normalization_51_gamma_read_readvariableop6savev2_batch_normalization_51_beta_read_readvariableop=savev2_batch_normalization_51_moving_mean_read_readvariableopAsavev2_batch_normalization_51_moving_variance_read_readvariableop5savev2_conv2d_transpose_10_kernel_read_readvariableop3savev2_conv2d_transpose_10_bias_read_readvariableop+savev2_conv2d_56_kernel_read_readvariableop)savev2_conv2d_56_bias_read_readvariableop7savev2_batch_normalization_53_gamma_read_readvariableop6savev2_batch_normalization_53_beta_read_readvariableop=savev2_batch_normalization_53_moving_mean_read_readvariableopAsavev2_batch_normalization_53_moving_variance_read_readvariableop5savev2_conv2d_transpose_11_kernel_read_readvariableop3savev2_conv2d_transpose_11_bias_read_readvariableop+savev2_conv2d_58_kernel_read_readvariableop)savev2_conv2d_58_bias_read_readvariableop7savev2_batch_normalization_55_gamma_read_readvariableop6savev2_batch_normalization_55_beta_read_readvariableop=savev2_batch_normalization_55_moving_mean_read_readvariableopAsavev2_batch_normalization_55_moving_variance_read_readvariableop+savev2_conv2d_59_kernel_read_readvariableop)savev2_conv2d_59_bias_read_readvariableop+savev2_cond_1_adam_iter_read_readvariableop-savev2_cond_1_adam_beta_1_read_readvariableop-savev2_cond_1_adam_beta_2_read_readvariableop,savev2_cond_1_adam_decay_read_readvariableop4savev2_cond_1_adam_learning_rate_read_readvariableop-savev2_current_loss_scale_read_readvariableop%savev2_good_steps_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop9savev2_cond_1_adam_conv2d_46_kernel_m_read_readvariableop7savev2_cond_1_adam_conv2d_46_bias_m_read_readvariableopEsavev2_cond_1_adam_batch_normalization_43_gamma_m_read_readvariableopDsavev2_cond_1_adam_batch_normalization_43_beta_m_read_readvariableop9savev2_cond_1_adam_conv2d_48_kernel_m_read_readvariableop7savev2_cond_1_adam_conv2d_48_bias_m_read_readvariableopEsavev2_cond_1_adam_batch_normalization_45_gamma_m_read_readvariableopDsavev2_cond_1_adam_batch_normalization_45_beta_m_read_readvariableop9savev2_cond_1_adam_conv2d_50_kernel_m_read_readvariableop7savev2_cond_1_adam_conv2d_50_bias_m_read_readvariableopEsavev2_cond_1_adam_batch_normalization_47_gamma_m_read_readvariableopDsavev2_cond_1_adam_batch_normalization_47_beta_m_read_readvariableop9savev2_cond_1_adam_conv2d_52_kernel_m_read_readvariableop7savev2_cond_1_adam_conv2d_52_bias_m_read_readvariableopEsavev2_cond_1_adam_batch_normalization_49_gamma_m_read_readvariableopDsavev2_cond_1_adam_batch_normalization_49_beta_m_read_readvariableopBsavev2_cond_1_adam_conv2d_transpose_9_kernel_m_read_readvariableop@savev2_cond_1_adam_conv2d_transpose_9_bias_m_read_readvariableop9savev2_cond_1_adam_conv2d_54_kernel_m_read_readvariableop7savev2_cond_1_adam_conv2d_54_bias_m_read_readvariableopEsavev2_cond_1_adam_batch_normalization_51_gamma_m_read_readvariableopDsavev2_cond_1_adam_batch_normalization_51_beta_m_read_readvariableopCsavev2_cond_1_adam_conv2d_transpose_10_kernel_m_read_readvariableopAsavev2_cond_1_adam_conv2d_transpose_10_bias_m_read_readvariableop9savev2_cond_1_adam_conv2d_56_kernel_m_read_readvariableop7savev2_cond_1_adam_conv2d_56_bias_m_read_readvariableopEsavev2_cond_1_adam_batch_normalization_53_gamma_m_read_readvariableopDsavev2_cond_1_adam_batch_normalization_53_beta_m_read_readvariableopCsavev2_cond_1_adam_conv2d_transpose_11_kernel_m_read_readvariableopAsavev2_cond_1_adam_conv2d_transpose_11_bias_m_read_readvariableop9savev2_cond_1_adam_conv2d_58_kernel_m_read_readvariableop7savev2_cond_1_adam_conv2d_58_bias_m_read_readvariableopEsavev2_cond_1_adam_batch_normalization_55_gamma_m_read_readvariableopDsavev2_cond_1_adam_batch_normalization_55_beta_m_read_readvariableop9savev2_cond_1_adam_conv2d_59_kernel_m_read_readvariableop7savev2_cond_1_adam_conv2d_59_bias_m_read_readvariableop9savev2_cond_1_adam_conv2d_46_kernel_v_read_readvariableop7savev2_cond_1_adam_conv2d_46_bias_v_read_readvariableopEsavev2_cond_1_adam_batch_normalization_43_gamma_v_read_readvariableopDsavev2_cond_1_adam_batch_normalization_43_beta_v_read_readvariableop9savev2_cond_1_adam_conv2d_48_kernel_v_read_readvariableop7savev2_cond_1_adam_conv2d_48_bias_v_read_readvariableopEsavev2_cond_1_adam_batch_normalization_45_gamma_v_read_readvariableopDsavev2_cond_1_adam_batch_normalization_45_beta_v_read_readvariableop9savev2_cond_1_adam_conv2d_50_kernel_v_read_readvariableop7savev2_cond_1_adam_conv2d_50_bias_v_read_readvariableopEsavev2_cond_1_adam_batch_normalization_47_gamma_v_read_readvariableopDsavev2_cond_1_adam_batch_normalization_47_beta_v_read_readvariableop9savev2_cond_1_adam_conv2d_52_kernel_v_read_readvariableop7savev2_cond_1_adam_conv2d_52_bias_v_read_readvariableopEsavev2_cond_1_adam_batch_normalization_49_gamma_v_read_readvariableopDsavev2_cond_1_adam_batch_normalization_49_beta_v_read_readvariableopBsavev2_cond_1_adam_conv2d_transpose_9_kernel_v_read_readvariableop@savev2_cond_1_adam_conv2d_transpose_9_bias_v_read_readvariableop9savev2_cond_1_adam_conv2d_54_kernel_v_read_readvariableop7savev2_cond_1_adam_conv2d_54_bias_v_read_readvariableopEsavev2_cond_1_adam_batch_normalization_51_gamma_v_read_readvariableopDsavev2_cond_1_adam_batch_normalization_51_beta_v_read_readvariableopCsavev2_cond_1_adam_conv2d_transpose_10_kernel_v_read_readvariableopAsavev2_cond_1_adam_conv2d_transpose_10_bias_v_read_readvariableop9savev2_cond_1_adam_conv2d_56_kernel_v_read_readvariableop7savev2_cond_1_adam_conv2d_56_bias_v_read_readvariableopEsavev2_cond_1_adam_batch_normalization_53_gamma_v_read_readvariableopDsavev2_cond_1_adam_batch_normalization_53_beta_v_read_readvariableopCsavev2_cond_1_adam_conv2d_transpose_11_kernel_v_read_readvariableopAsavev2_cond_1_adam_conv2d_transpose_11_bias_v_read_readvariableop9savev2_cond_1_adam_conv2d_58_kernel_v_read_readvariableop7savev2_cond_1_adam_conv2d_58_bias_v_read_readvariableopEsavev2_cond_1_adam_batch_normalization_55_gamma_v_read_readvariableopDsavev2_cond_1_adam_batch_normalization_55_beta_v_read_readvariableop9savev2_cond_1_adam_conv2d_59_kernel_v_read_readvariableop7savev2_cond_1_adam_conv2d_59_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
?
d
H__inference_activation_43_layer_call_and_return_conditional_losses_77338

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
?
s
I__inference_concatenate_10_layer_call_and_return_conditional_losses_77958

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
?
?
Q__inference_batch_normalization_53_layer_call_and_return_conditional_losses_77035

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
?
u
I__inference_concatenate_10_layer_call_and_return_conditional_losses_80753
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
?
?
6__inference_batch_normalization_55_layer_call_fn_81121

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
Q__inference_batch_normalization_55_layer_call_and_return_conditional_losses_771852
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
6__inference_batch_normalization_43_layer_call_fn_79902

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
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_763872
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
?
?
Q__inference_batch_normalization_47_layer_call_and_return_conditional_losses_76650

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
?
c
E__inference_dropout_20_layer_call_and_return_conditional_losses_80378

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
?
d
E__inference_dropout_20_layer_call_and_return_conditional_losses_80373

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
?
c
E__inference_dropout_21_layer_call_and_return_conditional_losses_77819

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
??
?
B__inference_model_3_layer_call_and_return_conditional_losses_78306
input_4
conv2d_46_77255
conv2d_46_77257 
batch_normalization_43_77324 
batch_normalization_43_77326 
batch_normalization_43_77328 
batch_normalization_43_77330
conv2d_48_77400
conv2d_48_77402 
batch_normalization_45_77469 
batch_normalization_45_77471 
batch_normalization_45_77473 
batch_normalization_45_77475
conv2d_50_77545
conv2d_50_77547 
batch_normalization_47_77614 
batch_normalization_47_77616 
batch_normalization_47_77618 
batch_normalization_47_77620
conv2d_52_77690
conv2d_52_77692 
batch_normalization_49_77759 
batch_normalization_49_77761 
batch_normalization_49_77763 
batch_normalization_49_77765
conv2d_transpose_9_77781
conv2d_transpose_9_77783
conv2d_54_77855
conv2d_54_77857 
batch_normalization_51_77924 
batch_normalization_51_77926 
batch_normalization_51_77928 
batch_normalization_51_77930
conv2d_transpose_10_77946
conv2d_transpose_10_77948
conv2d_56_78020
conv2d_56_78022 
batch_normalization_53_78089 
batch_normalization_53_78091 
batch_normalization_53_78093 
batch_normalization_53_78095
conv2d_transpose_11_78111
conv2d_transpose_11_78113
conv2d_58_78185
conv2d_58_78187 
batch_normalization_55_78254 
batch_normalization_55_78256 
batch_normalization_55_78258 
batch_normalization_55_78260
conv2d_59_78300
conv2d_59_78302
identity??.batch_normalization_43/StatefulPartitionedCall?.batch_normalization_45/StatefulPartitionedCall?.batch_normalization_47/StatefulPartitionedCall?.batch_normalization_49/StatefulPartitionedCall?.batch_normalization_51/StatefulPartitionedCall?.batch_normalization_53/StatefulPartitionedCall?.batch_normalization_55/StatefulPartitionedCall?!conv2d_46/StatefulPartitionedCall?!conv2d_48/StatefulPartitionedCall?!conv2d_50/StatefulPartitionedCall?!conv2d_52/StatefulPartitionedCall?!conv2d_54/StatefulPartitionedCall?!conv2d_56/StatefulPartitionedCall?!conv2d_58/StatefulPartitionedCall?!conv2d_59/StatefulPartitionedCall?+conv2d_transpose_10/StatefulPartitionedCall?+conv2d_transpose_11/StatefulPartitionedCall?*conv2d_transpose_9/StatefulPartitionedCall?"dropout_18/StatefulPartitionedCall?"dropout_19/StatefulPartitionedCall?"dropout_20/StatefulPartitionedCall?"dropout_21/StatefulPartitionedCall?"dropout_22/StatefulPartitionedCall?"dropout_23/StatefulPartitionedCallh
CastCastinput_4*

DstT0*

SrcT0*1
_output_shapes
:???????????2
Cast?
!conv2d_46/StatefulPartitionedCallStatefulPartitionedCallCast:y:0conv2d_46_77255conv2d_46_77257*
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
D__inference_conv2d_46_layer_call_and_return_conditional_losses_772442#
!conv2d_46/StatefulPartitionedCall?
.batch_normalization_43/StatefulPartitionedCallStatefulPartitionedCall*conv2d_46/StatefulPartitionedCall:output:0batch_normalization_43_77324batch_normalization_43_77326batch_normalization_43_77328batch_normalization_43_77330*
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
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_7727920
.batch_normalization_43/StatefulPartitionedCall?
activation_43/PartitionedCallPartitionedCall7batch_normalization_43/StatefulPartitionedCall:output:0*
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
H__inference_activation_43_layer_call_and_return_conditional_losses_773382
activation_43/PartitionedCall?
max_pooling2d_9/PartitionedCallPartitionedCall&activation_43/PartitionedCall:output:0*
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
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_764352!
max_pooling2d_9/PartitionedCall?
"dropout_18/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_9/PartitionedCall:output:0*
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
  ?E8? *N
fIRG
E__inference_dropout_18_layer_call_and_return_conditional_losses_773592$
"dropout_18/StatefulPartitionedCall?
!conv2d_48/StatefulPartitionedCallStatefulPartitionedCall+dropout_18/StatefulPartitionedCall:output:0conv2d_48_77400conv2d_48_77402*
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
D__inference_conv2d_48_layer_call_and_return_conditional_losses_773892#
!conv2d_48/StatefulPartitionedCall?
.batch_normalization_45/StatefulPartitionedCallStatefulPartitionedCall*conv2d_48/StatefulPartitionedCall:output:0batch_normalization_45_77469batch_normalization_45_77471batch_normalization_45_77473batch_normalization_45_77475*
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
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_7742420
.batch_normalization_45/StatefulPartitionedCall?
activation_45/PartitionedCallPartitionedCall7batch_normalization_45/StatefulPartitionedCall:output:0*
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
H__inference_activation_45_layer_call_and_return_conditional_losses_774832
activation_45/PartitionedCall?
 max_pooling2d_10/PartitionedCallPartitionedCall&activation_45/PartitionedCall:output:0*
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
  ?E8? *T
fORM
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_765512"
 max_pooling2d_10/PartitionedCall?
"dropout_19/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_10/PartitionedCall:output:0#^dropout_18/StatefulPartitionedCall*
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
  ?E8? *N
fIRG
E__inference_dropout_19_layer_call_and_return_conditional_losses_775042$
"dropout_19/StatefulPartitionedCall?
!conv2d_50/StatefulPartitionedCallStatefulPartitionedCall+dropout_19/StatefulPartitionedCall:output:0conv2d_50_77545conv2d_50_77547*
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
D__inference_conv2d_50_layer_call_and_return_conditional_losses_775342#
!conv2d_50/StatefulPartitionedCall?
.batch_normalization_47/StatefulPartitionedCallStatefulPartitionedCall*conv2d_50/StatefulPartitionedCall:output:0batch_normalization_47_77614batch_normalization_47_77616batch_normalization_47_77618batch_normalization_47_77620*
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
Q__inference_batch_normalization_47_layer_call_and_return_conditional_losses_7756920
.batch_normalization_47/StatefulPartitionedCall?
activation_47/PartitionedCallPartitionedCall7batch_normalization_47/StatefulPartitionedCall:output:0*
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
H__inference_activation_47_layer_call_and_return_conditional_losses_776282
activation_47/PartitionedCall?
 max_pooling2d_11/PartitionedCallPartitionedCall&activation_47/PartitionedCall:output:0*
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
  ?E8? *T
fORM
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_766672"
 max_pooling2d_11/PartitionedCall?
"dropout_20/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_11/PartitionedCall:output:0#^dropout_19/StatefulPartitionedCall*
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
  ?E8? *N
fIRG
E__inference_dropout_20_layer_call_and_return_conditional_losses_776492$
"dropout_20/StatefulPartitionedCall?
!conv2d_52/StatefulPartitionedCallStatefulPartitionedCall+dropout_20/StatefulPartitionedCall:output:0conv2d_52_77690conv2d_52_77692*
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
D__inference_conv2d_52_layer_call_and_return_conditional_losses_776792#
!conv2d_52/StatefulPartitionedCall?
.batch_normalization_49/StatefulPartitionedCallStatefulPartitionedCall*conv2d_52/StatefulPartitionedCall:output:0batch_normalization_49_77759batch_normalization_49_77761batch_normalization_49_77763batch_normalization_49_77765*
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
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_7771420
.batch_normalization_49/StatefulPartitionedCall?
activation_49/PartitionedCallPartitionedCall7batch_normalization_49/StatefulPartitionedCall:output:0*
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
H__inference_activation_49_layer_call_and_return_conditional_losses_777732
activation_49/PartitionedCall?
*conv2d_transpose_9/StatefulPartitionedCallStatefulPartitionedCall&activation_49/PartitionedCall:output:0conv2d_transpose_9_77781conv2d_transpose_9_77783*
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
M__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_768132,
*conv2d_transpose_9/StatefulPartitionedCall?
concatenate_9/PartitionedCallPartitionedCall3conv2d_transpose_9/StatefulPartitionedCall:output:0&activation_47/PartitionedCall:output:0*
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
H__inference_concatenate_9_layer_call_and_return_conditional_losses_777932
concatenate_9/PartitionedCall?
"dropout_21/StatefulPartitionedCallStatefulPartitionedCall&concatenate_9/PartitionedCall:output:0#^dropout_20/StatefulPartitionedCall*
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
  ?E8? *N
fIRG
E__inference_dropout_21_layer_call_and_return_conditional_losses_778142$
"dropout_21/StatefulPartitionedCall?
!conv2d_54/StatefulPartitionedCallStatefulPartitionedCall+dropout_21/StatefulPartitionedCall:output:0conv2d_54_77855conv2d_54_77857*
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
D__inference_conv2d_54_layer_call_and_return_conditional_losses_778442#
!conv2d_54/StatefulPartitionedCall?
.batch_normalization_51/StatefulPartitionedCallStatefulPartitionedCall*conv2d_54/StatefulPartitionedCall:output:0batch_normalization_51_77924batch_normalization_51_77926batch_normalization_51_77928batch_normalization_51_77930*
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
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_7787920
.batch_normalization_51/StatefulPartitionedCall?
activation_51/PartitionedCallPartitionedCall7batch_normalization_51/StatefulPartitionedCall:output:0*
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
H__inference_activation_51_layer_call_and_return_conditional_losses_779382
activation_51/PartitionedCall?
+conv2d_transpose_10/StatefulPartitionedCallStatefulPartitionedCall&activation_51/PartitionedCall:output:0conv2d_transpose_10_77946conv2d_transpose_10_77948*
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
  ?E8? *W
fRRP
N__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_769632-
+conv2d_transpose_10/StatefulPartitionedCall?
concatenate_10/PartitionedCallPartitionedCall4conv2d_transpose_10/StatefulPartitionedCall:output:0&activation_45/PartitionedCall:output:0*
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
  ?E8? *R
fMRK
I__inference_concatenate_10_layer_call_and_return_conditional_losses_779582 
concatenate_10/PartitionedCall?
"dropout_22/StatefulPartitionedCallStatefulPartitionedCall'concatenate_10/PartitionedCall:output:0#^dropout_21/StatefulPartitionedCall*
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
E__inference_dropout_22_layer_call_and_return_conditional_losses_779792$
"dropout_22/StatefulPartitionedCall?
!conv2d_56/StatefulPartitionedCallStatefulPartitionedCall+dropout_22/StatefulPartitionedCall:output:0conv2d_56_78020conv2d_56_78022*
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
D__inference_conv2d_56_layer_call_and_return_conditional_losses_780092#
!conv2d_56/StatefulPartitionedCall?
.batch_normalization_53/StatefulPartitionedCallStatefulPartitionedCall*conv2d_56/StatefulPartitionedCall:output:0batch_normalization_53_78089batch_normalization_53_78091batch_normalization_53_78093batch_normalization_53_78095*
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
Q__inference_batch_normalization_53_layer_call_and_return_conditional_losses_7804420
.batch_normalization_53/StatefulPartitionedCall?
activation_53/PartitionedCallPartitionedCall7batch_normalization_53/StatefulPartitionedCall:output:0*
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
H__inference_activation_53_layer_call_and_return_conditional_losses_781032
activation_53/PartitionedCall?
+conv2d_transpose_11/StatefulPartitionedCallStatefulPartitionedCall&activation_53/PartitionedCall:output:0conv2d_transpose_11_78111conv2d_transpose_11_78113*
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
  ?E8? *W
fRRP
N__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_771132-
+conv2d_transpose_11/StatefulPartitionedCall?
concatenate_11/PartitionedCallPartitionedCall4conv2d_transpose_11/StatefulPartitionedCall:output:0&activation_43/PartitionedCall:output:0*
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
  ?E8? *R
fMRK
I__inference_concatenate_11_layer_call_and_return_conditional_losses_781232 
concatenate_11/PartitionedCall?
"dropout_23/StatefulPartitionedCallStatefulPartitionedCall'concatenate_11/PartitionedCall:output:0#^dropout_22/StatefulPartitionedCall*
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
E__inference_dropout_23_layer_call_and_return_conditional_losses_781442$
"dropout_23/StatefulPartitionedCall?
!conv2d_58/StatefulPartitionedCallStatefulPartitionedCall+dropout_23/StatefulPartitionedCall:output:0conv2d_58_78185conv2d_58_78187*
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
D__inference_conv2d_58_layer_call_and_return_conditional_losses_781742#
!conv2d_58/StatefulPartitionedCall?
.batch_normalization_55/StatefulPartitionedCallStatefulPartitionedCall*conv2d_58/StatefulPartitionedCall:output:0batch_normalization_55_78254batch_normalization_55_78256batch_normalization_55_78258batch_normalization_55_78260*
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
Q__inference_batch_normalization_55_layer_call_and_return_conditional_losses_7820920
.batch_normalization_55/StatefulPartitionedCall?
activation_55/PartitionedCallPartitionedCall7batch_normalization_55/StatefulPartitionedCall:output:0*
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
H__inference_activation_55_layer_call_and_return_conditional_losses_782682
activation_55/PartitionedCall?
!conv2d_59/StatefulPartitionedCallStatefulPartitionedCall&activation_55/PartitionedCall:output:0conv2d_59_78300conv2d_59_78302*
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
D__inference_conv2d_59_layer_call_and_return_conditional_losses_782892#
!conv2d_59/StatefulPartitionedCall?
IdentityIdentity*conv2d_59/StatefulPartitionedCall:output:0/^batch_normalization_43/StatefulPartitionedCall/^batch_normalization_45/StatefulPartitionedCall/^batch_normalization_47/StatefulPartitionedCall/^batch_normalization_49/StatefulPartitionedCall/^batch_normalization_51/StatefulPartitionedCall/^batch_normalization_53/StatefulPartitionedCall/^batch_normalization_55/StatefulPartitionedCall"^conv2d_46/StatefulPartitionedCall"^conv2d_48/StatefulPartitionedCall"^conv2d_50/StatefulPartitionedCall"^conv2d_52/StatefulPartitionedCall"^conv2d_54/StatefulPartitionedCall"^conv2d_56/StatefulPartitionedCall"^conv2d_58/StatefulPartitionedCall"^conv2d_59/StatefulPartitionedCall,^conv2d_transpose_10/StatefulPartitionedCall,^conv2d_transpose_11/StatefulPartitionedCall+^conv2d_transpose_9/StatefulPartitionedCall#^dropout_18/StatefulPartitionedCall#^dropout_19/StatefulPartitionedCall#^dropout_20/StatefulPartitionedCall#^dropout_21/StatefulPartitionedCall#^dropout_22/StatefulPartitionedCall#^dropout_23/StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_43/StatefulPartitionedCall.batch_normalization_43/StatefulPartitionedCall2`
.batch_normalization_45/StatefulPartitionedCall.batch_normalization_45/StatefulPartitionedCall2`
.batch_normalization_47/StatefulPartitionedCall.batch_normalization_47/StatefulPartitionedCall2`
.batch_normalization_49/StatefulPartitionedCall.batch_normalization_49/StatefulPartitionedCall2`
.batch_normalization_51/StatefulPartitionedCall.batch_normalization_51/StatefulPartitionedCall2`
.batch_normalization_53/StatefulPartitionedCall.batch_normalization_53/StatefulPartitionedCall2`
.batch_normalization_55/StatefulPartitionedCall.batch_normalization_55/StatefulPartitionedCall2F
!conv2d_46/StatefulPartitionedCall!conv2d_46/StatefulPartitionedCall2F
!conv2d_48/StatefulPartitionedCall!conv2d_48/StatefulPartitionedCall2F
!conv2d_50/StatefulPartitionedCall!conv2d_50/StatefulPartitionedCall2F
!conv2d_52/StatefulPartitionedCall!conv2d_52/StatefulPartitionedCall2F
!conv2d_54/StatefulPartitionedCall!conv2d_54/StatefulPartitionedCall2F
!conv2d_56/StatefulPartitionedCall!conv2d_56/StatefulPartitionedCall2F
!conv2d_58/StatefulPartitionedCall!conv2d_58/StatefulPartitionedCall2F
!conv2d_59/StatefulPartitionedCall!conv2d_59/StatefulPartitionedCall2Z
+conv2d_transpose_10/StatefulPartitionedCall+conv2d_transpose_10/StatefulPartitionedCall2Z
+conv2d_transpose_11/StatefulPartitionedCall+conv2d_transpose_11/StatefulPartitionedCall2X
*conv2d_transpose_9/StatefulPartitionedCall*conv2d_transpose_9/StatefulPartitionedCall2H
"dropout_18/StatefulPartitionedCall"dropout_18/StatefulPartitionedCall2H
"dropout_19/StatefulPartitionedCall"dropout_19/StatefulPartitionedCall2H
"dropout_20/StatefulPartitionedCall"dropout_20/StatefulPartitionedCall2H
"dropout_21/StatefulPartitionedCall"dropout_21/StatefulPartitionedCall2H
"dropout_22/StatefulPartitionedCall"dropout_22/StatefulPartitionedCall2H
"dropout_23/StatefulPartitionedCall"dropout_23/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_4
?
?
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_80447

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
?
d
E__inference_dropout_21_layer_call_and_return_conditional_losses_77814

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
D__inference_conv2d_52_layer_call_and_return_conditional_losses_80400

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
?%
?
M__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_76813

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
?
d
H__inference_activation_47_layer_call_and_return_conditional_losses_80356

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
Q__inference_batch_normalization_55_layer_call_and_return_conditional_losses_77216

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
d
H__inference_activation_43_layer_call_and_return_conditional_losses_79984

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
?%
?
N__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_76963

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
?
d
H__inference_activation_51_layer_call_and_return_conditional_losses_77938

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
??
?
B__inference_model_3_layer_call_and_return_conditional_losses_78840

inputs
conv2d_46_78702
conv2d_46_78704 
batch_normalization_43_78707 
batch_normalization_43_78709 
batch_normalization_43_78711 
batch_normalization_43_78713
conv2d_48_78719
conv2d_48_78721 
batch_normalization_45_78724 
batch_normalization_45_78726 
batch_normalization_45_78728 
batch_normalization_45_78730
conv2d_50_78736
conv2d_50_78738 
batch_normalization_47_78741 
batch_normalization_47_78743 
batch_normalization_47_78745 
batch_normalization_47_78747
conv2d_52_78753
conv2d_52_78755 
batch_normalization_49_78758 
batch_normalization_49_78760 
batch_normalization_49_78762 
batch_normalization_49_78764
conv2d_transpose_9_78768
conv2d_transpose_9_78770
conv2d_54_78775
conv2d_54_78777 
batch_normalization_51_78780 
batch_normalization_51_78782 
batch_normalization_51_78784 
batch_normalization_51_78786
conv2d_transpose_10_78790
conv2d_transpose_10_78792
conv2d_56_78797
conv2d_56_78799 
batch_normalization_53_78802 
batch_normalization_53_78804 
batch_normalization_53_78806 
batch_normalization_53_78808
conv2d_transpose_11_78812
conv2d_transpose_11_78814
conv2d_58_78819
conv2d_58_78821 
batch_normalization_55_78824 
batch_normalization_55_78826 
batch_normalization_55_78828 
batch_normalization_55_78830
conv2d_59_78834
conv2d_59_78836
identity??.batch_normalization_43/StatefulPartitionedCall?.batch_normalization_45/StatefulPartitionedCall?.batch_normalization_47/StatefulPartitionedCall?.batch_normalization_49/StatefulPartitionedCall?.batch_normalization_51/StatefulPartitionedCall?.batch_normalization_53/StatefulPartitionedCall?.batch_normalization_55/StatefulPartitionedCall?!conv2d_46/StatefulPartitionedCall?!conv2d_48/StatefulPartitionedCall?!conv2d_50/StatefulPartitionedCall?!conv2d_52/StatefulPartitionedCall?!conv2d_54/StatefulPartitionedCall?!conv2d_56/StatefulPartitionedCall?!conv2d_58/StatefulPartitionedCall?!conv2d_59/StatefulPartitionedCall?+conv2d_transpose_10/StatefulPartitionedCall?+conv2d_transpose_11/StatefulPartitionedCall?*conv2d_transpose_9/StatefulPartitionedCallg
CastCastinputs*

DstT0*

SrcT0*1
_output_shapes
:???????????2
Cast?
!conv2d_46/StatefulPartitionedCallStatefulPartitionedCallCast:y:0conv2d_46_78702conv2d_46_78704*
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
D__inference_conv2d_46_layer_call_and_return_conditional_losses_772442#
!conv2d_46/StatefulPartitionedCall?
.batch_normalization_43/StatefulPartitionedCallStatefulPartitionedCall*conv2d_46/StatefulPartitionedCall:output:0batch_normalization_43_78707batch_normalization_43_78709batch_normalization_43_78711batch_normalization_43_78713*
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
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_7729720
.batch_normalization_43/StatefulPartitionedCall?
activation_43/PartitionedCallPartitionedCall7batch_normalization_43/StatefulPartitionedCall:output:0*
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
H__inference_activation_43_layer_call_and_return_conditional_losses_773382
activation_43/PartitionedCall?
max_pooling2d_9/PartitionedCallPartitionedCall&activation_43/PartitionedCall:output:0*
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
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_764352!
max_pooling2d_9/PartitionedCall?
dropout_18/PartitionedCallPartitionedCall(max_pooling2d_9/PartitionedCall:output:0*
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
  ?E8? *N
fIRG
E__inference_dropout_18_layer_call_and_return_conditional_losses_773642
dropout_18/PartitionedCall?
!conv2d_48/StatefulPartitionedCallStatefulPartitionedCall#dropout_18/PartitionedCall:output:0conv2d_48_78719conv2d_48_78721*
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
D__inference_conv2d_48_layer_call_and_return_conditional_losses_773892#
!conv2d_48/StatefulPartitionedCall?
.batch_normalization_45/StatefulPartitionedCallStatefulPartitionedCall*conv2d_48/StatefulPartitionedCall:output:0batch_normalization_45_78724batch_normalization_45_78726batch_normalization_45_78728batch_normalization_45_78730*
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
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_7744220
.batch_normalization_45/StatefulPartitionedCall?
activation_45/PartitionedCallPartitionedCall7batch_normalization_45/StatefulPartitionedCall:output:0*
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
H__inference_activation_45_layer_call_and_return_conditional_losses_774832
activation_45/PartitionedCall?
 max_pooling2d_10/PartitionedCallPartitionedCall&activation_45/PartitionedCall:output:0*
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
  ?E8? *T
fORM
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_765512"
 max_pooling2d_10/PartitionedCall?
dropout_19/PartitionedCallPartitionedCall)max_pooling2d_10/PartitionedCall:output:0*
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
  ?E8? *N
fIRG
E__inference_dropout_19_layer_call_and_return_conditional_losses_775092
dropout_19/PartitionedCall?
!conv2d_50/StatefulPartitionedCallStatefulPartitionedCall#dropout_19/PartitionedCall:output:0conv2d_50_78736conv2d_50_78738*
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
D__inference_conv2d_50_layer_call_and_return_conditional_losses_775342#
!conv2d_50/StatefulPartitionedCall?
.batch_normalization_47/StatefulPartitionedCallStatefulPartitionedCall*conv2d_50/StatefulPartitionedCall:output:0batch_normalization_47_78741batch_normalization_47_78743batch_normalization_47_78745batch_normalization_47_78747*
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
Q__inference_batch_normalization_47_layer_call_and_return_conditional_losses_7758720
.batch_normalization_47/StatefulPartitionedCall?
activation_47/PartitionedCallPartitionedCall7batch_normalization_47/StatefulPartitionedCall:output:0*
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
H__inference_activation_47_layer_call_and_return_conditional_losses_776282
activation_47/PartitionedCall?
 max_pooling2d_11/PartitionedCallPartitionedCall&activation_47/PartitionedCall:output:0*
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
  ?E8? *T
fORM
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_766672"
 max_pooling2d_11/PartitionedCall?
dropout_20/PartitionedCallPartitionedCall)max_pooling2d_11/PartitionedCall:output:0*
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
  ?E8? *N
fIRG
E__inference_dropout_20_layer_call_and_return_conditional_losses_776542
dropout_20/PartitionedCall?
!conv2d_52/StatefulPartitionedCallStatefulPartitionedCall#dropout_20/PartitionedCall:output:0conv2d_52_78753conv2d_52_78755*
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
D__inference_conv2d_52_layer_call_and_return_conditional_losses_776792#
!conv2d_52/StatefulPartitionedCall?
.batch_normalization_49/StatefulPartitionedCallStatefulPartitionedCall*conv2d_52/StatefulPartitionedCall:output:0batch_normalization_49_78758batch_normalization_49_78760batch_normalization_49_78762batch_normalization_49_78764*
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
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_7773220
.batch_normalization_49/StatefulPartitionedCall?
activation_49/PartitionedCallPartitionedCall7batch_normalization_49/StatefulPartitionedCall:output:0*
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
H__inference_activation_49_layer_call_and_return_conditional_losses_777732
activation_49/PartitionedCall?
*conv2d_transpose_9/StatefulPartitionedCallStatefulPartitionedCall&activation_49/PartitionedCall:output:0conv2d_transpose_9_78768conv2d_transpose_9_78770*
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
M__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_768132,
*conv2d_transpose_9/StatefulPartitionedCall?
concatenate_9/PartitionedCallPartitionedCall3conv2d_transpose_9/StatefulPartitionedCall:output:0&activation_47/PartitionedCall:output:0*
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
H__inference_concatenate_9_layer_call_and_return_conditional_losses_777932
concatenate_9/PartitionedCall?
dropout_21/PartitionedCallPartitionedCall&concatenate_9/PartitionedCall:output:0*
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
  ?E8? *N
fIRG
E__inference_dropout_21_layer_call_and_return_conditional_losses_778192
dropout_21/PartitionedCall?
!conv2d_54/StatefulPartitionedCallStatefulPartitionedCall#dropout_21/PartitionedCall:output:0conv2d_54_78775conv2d_54_78777*
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
D__inference_conv2d_54_layer_call_and_return_conditional_losses_778442#
!conv2d_54/StatefulPartitionedCall?
.batch_normalization_51/StatefulPartitionedCallStatefulPartitionedCall*conv2d_54/StatefulPartitionedCall:output:0batch_normalization_51_78780batch_normalization_51_78782batch_normalization_51_78784batch_normalization_51_78786*
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
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_7789720
.batch_normalization_51/StatefulPartitionedCall?
activation_51/PartitionedCallPartitionedCall7batch_normalization_51/StatefulPartitionedCall:output:0*
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
H__inference_activation_51_layer_call_and_return_conditional_losses_779382
activation_51/PartitionedCall?
+conv2d_transpose_10/StatefulPartitionedCallStatefulPartitionedCall&activation_51/PartitionedCall:output:0conv2d_transpose_10_78790conv2d_transpose_10_78792*
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
  ?E8? *W
fRRP
N__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_769632-
+conv2d_transpose_10/StatefulPartitionedCall?
concatenate_10/PartitionedCallPartitionedCall4conv2d_transpose_10/StatefulPartitionedCall:output:0&activation_45/PartitionedCall:output:0*
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
  ?E8? *R
fMRK
I__inference_concatenate_10_layer_call_and_return_conditional_losses_779582 
concatenate_10/PartitionedCall?
dropout_22/PartitionedCallPartitionedCall'concatenate_10/PartitionedCall:output:0*
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
E__inference_dropout_22_layer_call_and_return_conditional_losses_779842
dropout_22/PartitionedCall?
!conv2d_56/StatefulPartitionedCallStatefulPartitionedCall#dropout_22/PartitionedCall:output:0conv2d_56_78797conv2d_56_78799*
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
D__inference_conv2d_56_layer_call_and_return_conditional_losses_780092#
!conv2d_56/StatefulPartitionedCall?
.batch_normalization_53/StatefulPartitionedCallStatefulPartitionedCall*conv2d_56/StatefulPartitionedCall:output:0batch_normalization_53_78802batch_normalization_53_78804batch_normalization_53_78806batch_normalization_53_78808*
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
Q__inference_batch_normalization_53_layer_call_and_return_conditional_losses_7806220
.batch_normalization_53/StatefulPartitionedCall?
activation_53/PartitionedCallPartitionedCall7batch_normalization_53/StatefulPartitionedCall:output:0*
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
H__inference_activation_53_layer_call_and_return_conditional_losses_781032
activation_53/PartitionedCall?
+conv2d_transpose_11/StatefulPartitionedCallStatefulPartitionedCall&activation_53/PartitionedCall:output:0conv2d_transpose_11_78812conv2d_transpose_11_78814*
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
  ?E8? *W
fRRP
N__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_771132-
+conv2d_transpose_11/StatefulPartitionedCall?
concatenate_11/PartitionedCallPartitionedCall4conv2d_transpose_11/StatefulPartitionedCall:output:0&activation_43/PartitionedCall:output:0*
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
  ?E8? *R
fMRK
I__inference_concatenate_11_layer_call_and_return_conditional_losses_781232 
concatenate_11/PartitionedCall?
dropout_23/PartitionedCallPartitionedCall'concatenate_11/PartitionedCall:output:0*
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
E__inference_dropout_23_layer_call_and_return_conditional_losses_781492
dropout_23/PartitionedCall?
!conv2d_58/StatefulPartitionedCallStatefulPartitionedCall#dropout_23/PartitionedCall:output:0conv2d_58_78819conv2d_58_78821*
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
D__inference_conv2d_58_layer_call_and_return_conditional_losses_781742#
!conv2d_58/StatefulPartitionedCall?
.batch_normalization_55/StatefulPartitionedCallStatefulPartitionedCall*conv2d_58/StatefulPartitionedCall:output:0batch_normalization_55_78824batch_normalization_55_78826batch_normalization_55_78828batch_normalization_55_78830*
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
Q__inference_batch_normalization_55_layer_call_and_return_conditional_losses_7822720
.batch_normalization_55/StatefulPartitionedCall?
activation_55/PartitionedCallPartitionedCall7batch_normalization_55/StatefulPartitionedCall:output:0*
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
H__inference_activation_55_layer_call_and_return_conditional_losses_782682
activation_55/PartitionedCall?
!conv2d_59/StatefulPartitionedCallStatefulPartitionedCall&activation_55/PartitionedCall:output:0conv2d_59_78834conv2d_59_78836*
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
D__inference_conv2d_59_layer_call_and_return_conditional_losses_782892#
!conv2d_59/StatefulPartitionedCall?
IdentityIdentity*conv2d_59/StatefulPartitionedCall:output:0/^batch_normalization_43/StatefulPartitionedCall/^batch_normalization_45/StatefulPartitionedCall/^batch_normalization_47/StatefulPartitionedCall/^batch_normalization_49/StatefulPartitionedCall/^batch_normalization_51/StatefulPartitionedCall/^batch_normalization_53/StatefulPartitionedCall/^batch_normalization_55/StatefulPartitionedCall"^conv2d_46/StatefulPartitionedCall"^conv2d_48/StatefulPartitionedCall"^conv2d_50/StatefulPartitionedCall"^conv2d_52/StatefulPartitionedCall"^conv2d_54/StatefulPartitionedCall"^conv2d_56/StatefulPartitionedCall"^conv2d_58/StatefulPartitionedCall"^conv2d_59/StatefulPartitionedCall,^conv2d_transpose_10/StatefulPartitionedCall,^conv2d_transpose_11/StatefulPartitionedCall+^conv2d_transpose_9/StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_43/StatefulPartitionedCall.batch_normalization_43/StatefulPartitionedCall2`
.batch_normalization_45/StatefulPartitionedCall.batch_normalization_45/StatefulPartitionedCall2`
.batch_normalization_47/StatefulPartitionedCall.batch_normalization_47/StatefulPartitionedCall2`
.batch_normalization_49/StatefulPartitionedCall.batch_normalization_49/StatefulPartitionedCall2`
.batch_normalization_51/StatefulPartitionedCall.batch_normalization_51/StatefulPartitionedCall2`
.batch_normalization_53/StatefulPartitionedCall.batch_normalization_53/StatefulPartitionedCall2`
.batch_normalization_55/StatefulPartitionedCall.batch_normalization_55/StatefulPartitionedCall2F
!conv2d_46/StatefulPartitionedCall!conv2d_46/StatefulPartitionedCall2F
!conv2d_48/StatefulPartitionedCall!conv2d_48/StatefulPartitionedCall2F
!conv2d_50/StatefulPartitionedCall!conv2d_50/StatefulPartitionedCall2F
!conv2d_52/StatefulPartitionedCall!conv2d_52/StatefulPartitionedCall2F
!conv2d_54/StatefulPartitionedCall!conv2d_54/StatefulPartitionedCall2F
!conv2d_56/StatefulPartitionedCall!conv2d_56/StatefulPartitionedCall2F
!conv2d_58/StatefulPartitionedCall!conv2d_58/StatefulPartitionedCall2F
!conv2d_59/StatefulPartitionedCall!conv2d_59/StatefulPartitionedCall2Z
+conv2d_transpose_10/StatefulPartitionedCall+conv2d_transpose_10/StatefulPartitionedCall2Z
+conv2d_transpose_11/StatefulPartitionedCall+conv2d_transpose_11/StatefulPartitionedCall2X
*conv2d_transpose_9/StatefulPartitionedCall*conv2d_transpose_9/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_53_layer_call_and_return_conditional_losses_80845

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
)__inference_conv2d_56_layer_call_fn_80807

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
D__inference_conv2d_56_layer_call_and_return_conditional_losses_780092
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
?
?
6__inference_batch_normalization_45_layer_call_fn_80088

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
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_774242
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
?
?
Q__inference_batch_normalization_55_layer_call_and_return_conditional_losses_81044

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
?
?
Q__inference_batch_normalization_55_layer_call_and_return_conditional_losses_77185

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
Q__inference_batch_normalization_47_layer_call_and_return_conditional_losses_80307

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
?
I
-__inference_activation_55_layer_call_fn_81144

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
H__inference_activation_55_layer_call_and_return_conditional_losses_782682
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
?
F
*__inference_dropout_18_layer_call_fn_80016

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
  ?E8? *N
fIRG
E__inference_dropout_18_layer_call_and_return_conditional_losses_773642
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
?
I
-__inference_activation_51_layer_call_fn_80746

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
H__inference_activation_51_layer_call_and_return_conditional_losses_779382
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
d
E__inference_dropout_23_layer_call_and_return_conditional_losses_80970

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
?
?
Q__inference_batch_normalization_47_layer_call_and_return_conditional_losses_80261

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
L
0__inference_max_pooling2d_11_layer_call_fn_76673

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
  ?E8? *T
fORM
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_766672
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
?
d
H__inference_activation_55_layer_call_and_return_conditional_losses_81139

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
?
c
*__inference_dropout_18_layer_call_fn_80011

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
  ?E8? *N
fIRG
E__inference_dropout_18_layer_call_and_return_conditional_losses_773592
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
L
0__inference_max_pooling2d_10_layer_call_fn_76557

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
  ?E8? *T
fORM
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_765512
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
D__inference_conv2d_59_layer_call_and_return_conditional_losses_78289

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
g
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_76667

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
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_76766

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
?
?
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_80710

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
?
?
D__inference_conv2d_48_layer_call_and_return_conditional_losses_80028

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
?
u
I__inference_concatenate_11_layer_call_and_return_conditional_losses_80952
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
?
g
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_76551

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
?
?
'__inference_model_3_layer_call_fn_78696
input_4
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
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
B__inference_model_3_layer_call_and_return_conditional_losses_785932
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
_user_specified_name	input_4
?
c
E__inference_dropout_21_layer_call_and_return_conditional_losses_80577

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
?
?
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_79935

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
?
~
)__inference_conv2d_52_layer_call_fn_80409

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
D__inference_conv2d_52_layer_call_and_return_conditional_losses_776792
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
?
?
6__inference_batch_normalization_49_layer_call_fn_80473

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
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_777322
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
?
?
6__inference_batch_normalization_47_layer_call_fn_80287

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
Q__inference_batch_normalization_47_layer_call_and_return_conditional_losses_775872
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
?
c
E__inference_dropout_18_layer_call_and_return_conditional_losses_77364

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
?
?
D__inference_conv2d_50_layer_call_and_return_conditional_losses_80214

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
?
?
6__inference_batch_normalization_51_layer_call_fn_80659

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
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_768852
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
d
E__inference_dropout_19_layer_call_and_return_conditional_losses_77504

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
?
?
6__inference_batch_normalization_45_layer_call_fn_80152

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
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_765032
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
?
~
)__inference_conv2d_46_layer_call_fn_79851

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
D__inference_conv2d_46_layer_call_and_return_conditional_losses_772442
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
?
t
H__inference_concatenate_9_layer_call_and_return_conditional_losses_80554
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
?
I
-__inference_activation_53_layer_call_fn_80945

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
H__inference_activation_53_layer_call_and_return_conditional_losses_781032
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
?
?
6__inference_batch_normalization_51_layer_call_fn_80672

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
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_769162
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
?
c
E__inference_dropout_23_layer_call_and_return_conditional_losses_80975

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
?%
?
N__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_77113

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
d
E__inference_dropout_22_layer_call_and_return_conditional_losses_77979

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
?
?
#__inference_signature_wrapper_79058
input_4
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
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
 __inference__wrapped_model_763252
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
_user_specified_name	input_4
?
?
Q__inference_batch_normalization_53_layer_call_and_return_conditional_losses_77066

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
?
d
E__inference_dropout_20_layer_call_and_return_conditional_losses_77649

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
?
?
D__inference_conv2d_56_layer_call_and_return_conditional_losses_78009

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
?
d
H__inference_activation_49_layer_call_and_return_conditional_losses_77773

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
input_4:
serving_default_input_4:0???????????G
	conv2d_59:
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
?__call__"ټ
_tf_keras_network??{"class_name": "Functional", "name": "model_3", "trainable": true, "expects_training_arg": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 544, 736, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_46", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_46", "inbound_nodes": [[["input_4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_43", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_43", "inbound_nodes": [[["conv2d_46", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_43", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "activation": "relu"}, "name": "activation_43", "inbound_nodes": [[["batch_normalization_43", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_9", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_9", "inbound_nodes": [[["activation_43", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_18", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_18", "inbound_nodes": [[["max_pooling2d_9", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_48", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_48", "inbound_nodes": [[["dropout_18", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_45", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_45", "inbound_nodes": [[["conv2d_48", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_45", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "activation": "relu"}, "name": "activation_45", "inbound_nodes": [[["batch_normalization_45", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_10", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_10", "inbound_nodes": [[["activation_45", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_19", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_19", "inbound_nodes": [[["max_pooling2d_10", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_50", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_50", "inbound_nodes": [[["dropout_19", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_47", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_47", "inbound_nodes": [[["conv2d_50", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_47", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "activation": "relu"}, "name": "activation_47", "inbound_nodes": [[["batch_normalization_47", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_11", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_11", "inbound_nodes": [[["activation_47", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_20", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_20", "inbound_nodes": [[["max_pooling2d_11", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_52", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_52", "inbound_nodes": [[["dropout_20", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_49", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_49", "inbound_nodes": [[["conv2d_52", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_49", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "activation": "relu"}, "name": "activation_49", "inbound_nodes": [[["batch_normalization_49", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_9", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_9", "inbound_nodes": [[["activation_49", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_9", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": -1}, "name": "concatenate_9", "inbound_nodes": [[["conv2d_transpose_9", 0, 0, {}], ["activation_47", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_21", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_21", "inbound_nodes": [[["concatenate_9", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_54", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_54", "inbound_nodes": [[["dropout_21", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_51", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_51", "inbound_nodes": [[["conv2d_54", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_51", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "activation": "relu"}, "name": "activation_51", "inbound_nodes": [[["batch_normalization_51", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_10", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_10", "inbound_nodes": [[["activation_51", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_10", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": -1}, "name": "concatenate_10", "inbound_nodes": [[["conv2d_transpose_10", 0, 0, {}], ["activation_45", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_22", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_22", "inbound_nodes": [[["concatenate_10", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_56", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_56", "inbound_nodes": [[["dropout_22", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_53", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_53", "inbound_nodes": [[["conv2d_56", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_53", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "activation": "relu"}, "name": "activation_53", "inbound_nodes": [[["batch_normalization_53", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_11", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_11", "inbound_nodes": [[["activation_53", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_11", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": -1}, "name": "concatenate_11", "inbound_nodes": [[["conv2d_transpose_11", 0, 0, {}], ["activation_43", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_23", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_23", "inbound_nodes": [[["concatenate_11", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_58", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_58", "inbound_nodes": [[["dropout_23", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_55", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_55", "inbound_nodes": [[["conv2d_58", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_55", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "activation": "relu"}, "name": "activation_55", "inbound_nodes": [[["batch_normalization_55", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_59", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 7, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_59", "inbound_nodes": [[["activation_55", 0, 0, {}]]]}], "input_layers": [["input_4", 0, 0]], "output_layers": [["conv2d_59", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 544, 736, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 544, 736, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 544, 736, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_46", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_46", "inbound_nodes": [[["input_4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_43", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_43", "inbound_nodes": [[["conv2d_46", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_43", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "activation": "relu"}, "name": "activation_43", "inbound_nodes": [[["batch_normalization_43", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_9", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_9", "inbound_nodes": [[["activation_43", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_18", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_18", "inbound_nodes": [[["max_pooling2d_9", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_48", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_48", "inbound_nodes": [[["dropout_18", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_45", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_45", "inbound_nodes": [[["conv2d_48", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_45", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "activation": "relu"}, "name": "activation_45", "inbound_nodes": [[["batch_normalization_45", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_10", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_10", "inbound_nodes": [[["activation_45", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_19", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_19", "inbound_nodes": [[["max_pooling2d_10", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_50", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_50", "inbound_nodes": [[["dropout_19", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_47", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_47", "inbound_nodes": [[["conv2d_50", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_47", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "activation": "relu"}, "name": "activation_47", "inbound_nodes": [[["batch_normalization_47", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_11", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_11", "inbound_nodes": [[["activation_47", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_20", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_20", "inbound_nodes": [[["max_pooling2d_11", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_52", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_52", "inbound_nodes": [[["dropout_20", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_49", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_49", "inbound_nodes": [[["conv2d_52", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_49", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "activation": "relu"}, "name": "activation_49", "inbound_nodes": [[["batch_normalization_49", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_9", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_9", "inbound_nodes": [[["activation_49", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_9", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": -1}, "name": "concatenate_9", "inbound_nodes": [[["conv2d_transpose_9", 0, 0, {}], ["activation_47", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_21", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_21", "inbound_nodes": [[["concatenate_9", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_54", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_54", "inbound_nodes": [[["dropout_21", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_51", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_51", "inbound_nodes": [[["conv2d_54", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_51", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "activation": "relu"}, "name": "activation_51", "inbound_nodes": [[["batch_normalization_51", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_10", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_10", "inbound_nodes": [[["activation_51", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_10", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": -1}, "name": "concatenate_10", "inbound_nodes": [[["conv2d_transpose_10", 0, 0, {}], ["activation_45", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_22", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_22", "inbound_nodes": [[["concatenate_10", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_56", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_56", "inbound_nodes": [[["dropout_22", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_53", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_53", "inbound_nodes": [[["conv2d_56", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_53", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "activation": "relu"}, "name": "activation_53", "inbound_nodes": [[["batch_normalization_53", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_11", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_11", "inbound_nodes": [[["activation_53", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_11", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": -1}, "name": "concatenate_11", "inbound_nodes": [[["conv2d_transpose_11", 0, 0, {}], ["activation_43", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_23", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_23", "inbound_nodes": [[["concatenate_11", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_58", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_58", "inbound_nodes": [[["dropout_23", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_55", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_55", "inbound_nodes": [[["conv2d_58", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_55", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "activation": "relu"}, "name": "activation_55", "inbound_nodes": [[["batch_normalization_55", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_59", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 7, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_59", "inbound_nodes": [[["activation_55", 0, 0, {}]]]}], "input_layers": [["input_4", 0, 0]], "output_layers": [["conv2d_59", 0, 0]]}}, "training_config": {"loss": "mse", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "iou_coef", "dtype": "float32", "fn": "iou_coef"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "LossScaleOptimizer", "config": {"inner_optimizer": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}, "dynamic": true, "initial_scale": 32768.0, "dynamic_growth_steps": 2000}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_4", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 544, 736, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 544, 736, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}}
?


-kernel
.bias
/	variables
0trainable_variables
1regularization_losses
2	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_46", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_46", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 544, 736, 3]}}
?

3axis
	4gamma
5beta
6moving_mean
7moving_variance
8	variables
9trainable_variables
:regularization_losses
;	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_43", "trainable": true, "expects_training_arg": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_43", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 544, 736, 32]}}
?
<	variables
=trainable_variables
>regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_43", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_43", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "activation": "relu"}}
?
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_9", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_9", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_18", "trainable": true, "expects_training_arg": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_18", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "rate": 0.1, "noise_shape": null, "seed": null}}
?


Hkernel
Ibias
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_48", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_48", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 272, 368, 32]}}
?

Naxis
	Ogamma
Pbeta
Qmoving_mean
Rmoving_variance
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_45", "trainable": true, "expects_training_arg": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_45", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 272, 368, 64]}}
?
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_45", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_45", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "activation": "relu"}}
?
[	variables
\trainable_variables
]regularization_losses
^	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_10", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_10", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
_	variables
`trainable_variables
aregularization_losses
b	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_19", "trainable": true, "expects_training_arg": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_19", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "rate": 0.1, "noise_shape": null, "seed": null}}
?


ckernel
dbias
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_50", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_50", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 136, 184, 64]}}
?

iaxis
	jgamma
kbeta
lmoving_mean
mmoving_variance
n	variables
otrainable_variables
pregularization_losses
q	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_47", "trainable": true, "expects_training_arg": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_47", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 136, 184, 128]}}
?
r	variables
strainable_variables
tregularization_losses
u	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_47", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_47", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "activation": "relu"}}
?
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_11", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_11", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
z	variables
{trainable_variables
|regularization_losses
}	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_20", "trainable": true, "expects_training_arg": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_20", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "rate": 0.1, "noise_shape": null, "seed": null}}
?


~kernel
bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_52", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_52", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 68, 92, 128]}}
?

	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_49", "trainable": true, "expects_training_arg": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_49", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 68, 92, 256]}}
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_49", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_49", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "activation": "relu"}}
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_9", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose_9", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 68, 92, 256]}}
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate_9", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate_9", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 136, 184, 128]}, {"class_name": "TensorShape", "items": [null, 136, 184, 128]}]}
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_21", "trainable": true, "expects_training_arg": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_21", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "rate": 0.1, "noise_shape": null, "seed": null}}
?

?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_54", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_54", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 136, 184, 256]}}
?

	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_51", "trainable": true, "expects_training_arg": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_51", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 136, 184, 128]}}
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_51", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_51", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "activation": "relu"}}
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_10", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose_10", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 136, 184, 128]}}
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate_10", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate_10", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 272, 368, 64]}, {"class_name": "TensorShape", "items": [null, 272, 368, 64]}]}
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_22", "trainable": true, "expects_training_arg": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_22", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "rate": 0.1, "noise_shape": null, "seed": null}}
?

?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_56", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_56", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 272, 368, 128]}}
?

	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_53", "trainable": true, "expects_training_arg": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_53", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 272, 368, 64]}}
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_53", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_53", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "activation": "relu"}}
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_11", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose_11", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 272, 368, 64]}}
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate_11", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate_11", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 544, 736, 32]}, {"class_name": "TensorShape", "items": [null, 544, 736, 32]}]}
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_23", "trainable": true, "expects_training_arg": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_23", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "rate": 0.1, "noise_shape": null, "seed": null}}
?

?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_58", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_58", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 544, 736, 64]}}
?

	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_55", "trainable": true, "expects_training_arg": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_55", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 544, 736, 32]}}
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_55", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_55", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "activation": "relu"}}
?

?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_59", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_59", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 7, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 544, 736, 32]}}
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
?layer_metrics
?non_trainable_variables
?metrics
(	variables
 ?layer_regularization_losses
?layers
)trainable_variables
*regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
*:( 2conv2d_46/kernel
: 2conv2d_46/bias
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?non_trainable_variables
?metrics
/	variables
 ?layer_regularization_losses
?layers
0trainable_variables
1regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:( 2batch_normalization_43/gamma
):' 2batch_normalization_43/beta
2:0  (2"batch_normalization_43/moving_mean
6:4  (2&batch_normalization_43/moving_variance
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
 "
trackable_list_wrapper
?
?layer_metrics
?non_trainable_variables
?metrics
8	variables
 ?layer_regularization_losses
?layers
9trainable_variables
:regularization_losses
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
?layer_metrics
?non_trainable_variables
?metrics
<	variables
 ?layer_regularization_losses
?layers
=trainable_variables
>regularization_losses
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
?layer_metrics
?non_trainable_variables
?metrics
@	variables
 ?layer_regularization_losses
?layers
Atrainable_variables
Bregularization_losses
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
?layer_metrics
?non_trainable_variables
?metrics
D	variables
 ?layer_regularization_losses
?layers
Etrainable_variables
Fregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:( @2conv2d_48/kernel
:@2conv2d_48/bias
.
H0
I1"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?non_trainable_variables
?metrics
J	variables
 ?layer_regularization_losses
?layers
Ktrainable_variables
Lregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2batch_normalization_45/gamma
):'@2batch_normalization_45/beta
2:0@ (2"batch_normalization_45/moving_mean
6:4@ (2&batch_normalization_45/moving_variance
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
 "
trackable_list_wrapper
?
?layer_metrics
?non_trainable_variables
?metrics
S	variables
 ?layer_regularization_losses
?layers
Ttrainable_variables
Uregularization_losses
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
?layer_metrics
?non_trainable_variables
?metrics
W	variables
 ?layer_regularization_losses
?layers
Xtrainable_variables
Yregularization_losses
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
?layer_metrics
?non_trainable_variables
?metrics
[	variables
 ?layer_regularization_losses
?layers
\trainable_variables
]regularization_losses
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
?layer_metrics
?non_trainable_variables
?metrics
_	variables
 ?layer_regularization_losses
?layers
`trainable_variables
aregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)@?2conv2d_50/kernel
:?2conv2d_50/bias
.
c0
d1"
trackable_list_wrapper
.
c0
d1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?non_trainable_variables
?metrics
e	variables
 ?layer_regularization_losses
?layers
ftrainable_variables
gregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)?2batch_normalization_47/gamma
*:(?2batch_normalization_47/beta
3:1? (2"batch_normalization_47/moving_mean
7:5? (2&batch_normalization_47/moving_variance
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
 "
trackable_list_wrapper
?
?layer_metrics
?non_trainable_variables
?metrics
n	variables
 ?layer_regularization_losses
?layers
otrainable_variables
pregularization_losses
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
?layer_metrics
?non_trainable_variables
?metrics
r	variables
 ?layer_regularization_losses
?layers
strainable_variables
tregularization_losses
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
?layer_metrics
?non_trainable_variables
?metrics
v	variables
 ?layer_regularization_losses
?layers
wtrainable_variables
xregularization_losses
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
?layer_metrics
?non_trainable_variables
?metrics
z	variables
 ?layer_regularization_losses
?layers
{trainable_variables
|regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
,:*??2conv2d_52/kernel
:?2conv2d_52/bias
.
~0
1"
trackable_list_wrapper
.
~0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?non_trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)?2batch_normalization_49/gamma
*:(?2batch_normalization_49/beta
3:1? (2"batch_normalization_49/moving_mean
7:5? (2&batch_normalization_49/moving_variance
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
 "
trackable_list_wrapper
?
?layer_metrics
?non_trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
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
?layer_metrics
?non_trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
5:3??2conv2d_transpose_9/kernel
&:$?2conv2d_transpose_9/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?non_trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
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
?layer_metrics
?non_trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
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
?layer_metrics
?non_trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
,:*??2conv2d_54/kernel
:?2conv2d_54/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?non_trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)?2batch_normalization_51/gamma
*:(?2batch_normalization_51/beta
3:1? (2"batch_normalization_51/moving_mean
7:5? (2&batch_normalization_51/moving_variance
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
 "
trackable_list_wrapper
?
?layer_metrics
?non_trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
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
?layer_metrics
?non_trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
5:3@?2conv2d_transpose_10/kernel
&:$@2conv2d_transpose_10/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?non_trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
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
?layer_metrics
?non_trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
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
?layer_metrics
?non_trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)?@2conv2d_56/kernel
:@2conv2d_56/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?non_trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2batch_normalization_53/gamma
):'@2batch_normalization_53/beta
2:0@ (2"batch_normalization_53/moving_mean
6:4@ (2&batch_normalization_53/moving_variance
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
 "
trackable_list_wrapper
?
?layer_metrics
?non_trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
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
?layer_metrics
?non_trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
4:2 @2conv2d_transpose_11/kernel
&:$ 2conv2d_transpose_11/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?non_trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
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
?layer_metrics
?non_trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
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
?layer_metrics
?non_trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(@ 2conv2d_58/kernel
: 2conv2d_58/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?non_trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:( 2batch_normalization_55/gamma
):' 2batch_normalization_55/beta
2:0  (2"batch_normalization_55/moving_mean
6:4  (2&batch_normalization_55/moving_variance
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
 "
trackable_list_wrapper
?
?layer_metrics
?non_trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
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
?layer_metrics
?non_trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:( 2conv2d_59/kernel
:2conv2d_59/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?non_trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
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
6:4 2cond_1/Adam/conv2d_46/kernel/m
(:& 2cond_1/Adam/conv2d_46/bias/m
6:4 2*cond_1/Adam/batch_normalization_43/gamma/m
5:3 2)cond_1/Adam/batch_normalization_43/beta/m
6:4 @2cond_1/Adam/conv2d_48/kernel/m
(:&@2cond_1/Adam/conv2d_48/bias/m
6:4@2*cond_1/Adam/batch_normalization_45/gamma/m
5:3@2)cond_1/Adam/batch_normalization_45/beta/m
7:5@?2cond_1/Adam/conv2d_50/kernel/m
):'?2cond_1/Adam/conv2d_50/bias/m
7:5?2*cond_1/Adam/batch_normalization_47/gamma/m
6:4?2)cond_1/Adam/batch_normalization_47/beta/m
8:6??2cond_1/Adam/conv2d_52/kernel/m
):'?2cond_1/Adam/conv2d_52/bias/m
7:5?2*cond_1/Adam/batch_normalization_49/gamma/m
6:4?2)cond_1/Adam/batch_normalization_49/beta/m
A:???2'cond_1/Adam/conv2d_transpose_9/kernel/m
2:0?2%cond_1/Adam/conv2d_transpose_9/bias/m
8:6??2cond_1/Adam/conv2d_54/kernel/m
):'?2cond_1/Adam/conv2d_54/bias/m
7:5?2*cond_1/Adam/batch_normalization_51/gamma/m
6:4?2)cond_1/Adam/batch_normalization_51/beta/m
A:?@?2(cond_1/Adam/conv2d_transpose_10/kernel/m
2:0@2&cond_1/Adam/conv2d_transpose_10/bias/m
7:5?@2cond_1/Adam/conv2d_56/kernel/m
(:&@2cond_1/Adam/conv2d_56/bias/m
6:4@2*cond_1/Adam/batch_normalization_53/gamma/m
5:3@2)cond_1/Adam/batch_normalization_53/beta/m
@:> @2(cond_1/Adam/conv2d_transpose_11/kernel/m
2:0 2&cond_1/Adam/conv2d_transpose_11/bias/m
6:4@ 2cond_1/Adam/conv2d_58/kernel/m
(:& 2cond_1/Adam/conv2d_58/bias/m
6:4 2*cond_1/Adam/batch_normalization_55/gamma/m
5:3 2)cond_1/Adam/batch_normalization_55/beta/m
6:4 2cond_1/Adam/conv2d_59/kernel/m
(:&2cond_1/Adam/conv2d_59/bias/m
6:4 2cond_1/Adam/conv2d_46/kernel/v
(:& 2cond_1/Adam/conv2d_46/bias/v
6:4 2*cond_1/Adam/batch_normalization_43/gamma/v
5:3 2)cond_1/Adam/batch_normalization_43/beta/v
6:4 @2cond_1/Adam/conv2d_48/kernel/v
(:&@2cond_1/Adam/conv2d_48/bias/v
6:4@2*cond_1/Adam/batch_normalization_45/gamma/v
5:3@2)cond_1/Adam/batch_normalization_45/beta/v
7:5@?2cond_1/Adam/conv2d_50/kernel/v
):'?2cond_1/Adam/conv2d_50/bias/v
7:5?2*cond_1/Adam/batch_normalization_47/gamma/v
6:4?2)cond_1/Adam/batch_normalization_47/beta/v
8:6??2cond_1/Adam/conv2d_52/kernel/v
):'?2cond_1/Adam/conv2d_52/bias/v
7:5?2*cond_1/Adam/batch_normalization_49/gamma/v
6:4?2)cond_1/Adam/batch_normalization_49/beta/v
A:???2'cond_1/Adam/conv2d_transpose_9/kernel/v
2:0?2%cond_1/Adam/conv2d_transpose_9/bias/v
8:6??2cond_1/Adam/conv2d_54/kernel/v
):'?2cond_1/Adam/conv2d_54/bias/v
7:5?2*cond_1/Adam/batch_normalization_51/gamma/v
6:4?2)cond_1/Adam/batch_normalization_51/beta/v
A:?@?2(cond_1/Adam/conv2d_transpose_10/kernel/v
2:0@2&cond_1/Adam/conv2d_transpose_10/bias/v
7:5?@2cond_1/Adam/conv2d_56/kernel/v
(:&@2cond_1/Adam/conv2d_56/bias/v
6:4@2*cond_1/Adam/batch_normalization_53/gamma/v
5:3@2)cond_1/Adam/batch_normalization_53/beta/v
@:> @2(cond_1/Adam/conv2d_transpose_11/kernel/v
2:0 2&cond_1/Adam/conv2d_transpose_11/bias/v
6:4@ 2cond_1/Adam/conv2d_58/kernel/v
(:& 2cond_1/Adam/conv2d_58/bias/v
6:4 2*cond_1/Adam/batch_normalization_55/gamma/v
5:3 2)cond_1/Adam/batch_normalization_55/beta/v
6:4 2cond_1/Adam/conv2d_59/kernel/v
(:&2cond_1/Adam/conv2d_59/bias/v
?2?
B__inference_model_3_layer_call_and_return_conditional_losses_79367
B__inference_model_3_layer_call_and_return_conditional_losses_78306
B__inference_model_3_layer_call_and_return_conditional_losses_79620
B__inference_model_3_layer_call_and_return_conditional_losses_78448?
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
 __inference__wrapped_model_76325?
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
input_4???????????
?2?
'__inference_model_3_layer_call_fn_79725
'__inference_model_3_layer_call_fn_78696
'__inference_model_3_layer_call_fn_79830
'__inference_model_3_layer_call_fn_78943?
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
D__inference_conv2d_46_layer_call_and_return_conditional_losses_79842?
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
)__inference_conv2d_46_layer_call_fn_79851?
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
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_79889
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_79935
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_79871
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_79953?
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
6__inference_batch_normalization_43_layer_call_fn_79966
6__inference_batch_normalization_43_layer_call_fn_79979
6__inference_batch_normalization_43_layer_call_fn_79915
6__inference_batch_normalization_43_layer_call_fn_79902?
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
H__inference_activation_43_layer_call_and_return_conditional_losses_79984?
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
-__inference_activation_43_layer_call_fn_79989?
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
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_76435?
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
/__inference_max_pooling2d_9_layer_call_fn_76441?
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
E__inference_dropout_18_layer_call_and_return_conditional_losses_80006
E__inference_dropout_18_layer_call_and_return_conditional_losses_80001?
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
*__inference_dropout_18_layer_call_fn_80011
*__inference_dropout_18_layer_call_fn_80016?
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
D__inference_conv2d_48_layer_call_and_return_conditional_losses_80028?
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
)__inference_conv2d_48_layer_call_fn_80037?
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
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_80075
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_80139
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_80121
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_80057?
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
6__inference_batch_normalization_45_layer_call_fn_80152
6__inference_batch_normalization_45_layer_call_fn_80088
6__inference_batch_normalization_45_layer_call_fn_80165
6__inference_batch_normalization_45_layer_call_fn_80101?
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
H__inference_activation_45_layer_call_and_return_conditional_losses_80170?
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
-__inference_activation_45_layer_call_fn_80175?
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
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_76551?
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
0__inference_max_pooling2d_10_layer_call_fn_76557?
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
E__inference_dropout_19_layer_call_and_return_conditional_losses_80187
E__inference_dropout_19_layer_call_and_return_conditional_losses_80192?
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
*__inference_dropout_19_layer_call_fn_80197
*__inference_dropout_19_layer_call_fn_80202?
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
D__inference_conv2d_50_layer_call_and_return_conditional_losses_80214?
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
)__inference_conv2d_50_layer_call_fn_80223?
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
Q__inference_batch_normalization_47_layer_call_and_return_conditional_losses_80261
Q__inference_batch_normalization_47_layer_call_and_return_conditional_losses_80325
Q__inference_batch_normalization_47_layer_call_and_return_conditional_losses_80307
Q__inference_batch_normalization_47_layer_call_and_return_conditional_losses_80243?
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
6__inference_batch_normalization_47_layer_call_fn_80351
6__inference_batch_normalization_47_layer_call_fn_80274
6__inference_batch_normalization_47_layer_call_fn_80287
6__inference_batch_normalization_47_layer_call_fn_80338?
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
H__inference_activation_47_layer_call_and_return_conditional_losses_80356?
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
-__inference_activation_47_layer_call_fn_80361?
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
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_76667?
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
0__inference_max_pooling2d_11_layer_call_fn_76673?
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
E__inference_dropout_20_layer_call_and_return_conditional_losses_80373
E__inference_dropout_20_layer_call_and_return_conditional_losses_80378?
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
*__inference_dropout_20_layer_call_fn_80383
*__inference_dropout_20_layer_call_fn_80388?
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
D__inference_conv2d_52_layer_call_and_return_conditional_losses_80400?
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
)__inference_conv2d_52_layer_call_fn_80409?
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
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_80447
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_80429
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_80511
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_80493?
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
6__inference_batch_normalization_49_layer_call_fn_80460
6__inference_batch_normalization_49_layer_call_fn_80537
6__inference_batch_normalization_49_layer_call_fn_80524
6__inference_batch_normalization_49_layer_call_fn_80473?
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
H__inference_activation_49_layer_call_and_return_conditional_losses_80542?
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
-__inference_activation_49_layer_call_fn_80547?
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
M__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_76813?
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
2__inference_conv2d_transpose_9_layer_call_fn_76823?
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
H__inference_concatenate_9_layer_call_and_return_conditional_losses_80554?
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
-__inference_concatenate_9_layer_call_fn_80560?
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
E__inference_dropout_21_layer_call_and_return_conditional_losses_80577
E__inference_dropout_21_layer_call_and_return_conditional_losses_80572?
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
*__inference_dropout_21_layer_call_fn_80582
*__inference_dropout_21_layer_call_fn_80587?
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
D__inference_conv2d_54_layer_call_and_return_conditional_losses_80599?
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
)__inference_conv2d_54_layer_call_fn_80608?
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
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_80628
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_80646
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_80692
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_80710?
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
6__inference_batch_normalization_51_layer_call_fn_80659
6__inference_batch_normalization_51_layer_call_fn_80723
6__inference_batch_normalization_51_layer_call_fn_80672
6__inference_batch_normalization_51_layer_call_fn_80736?
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
H__inference_activation_51_layer_call_and_return_conditional_losses_80741?
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
-__inference_activation_51_layer_call_fn_80746?
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
N__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_76963?
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
3__inference_conv2d_transpose_10_layer_call_fn_76973?
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
I__inference_concatenate_10_layer_call_and_return_conditional_losses_80753?
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
.__inference_concatenate_10_layer_call_fn_80759?
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
E__inference_dropout_22_layer_call_and_return_conditional_losses_80776
E__inference_dropout_22_layer_call_and_return_conditional_losses_80771?
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
*__inference_dropout_22_layer_call_fn_80786
*__inference_dropout_22_layer_call_fn_80781?
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
D__inference_conv2d_56_layer_call_and_return_conditional_losses_80798?
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
)__inference_conv2d_56_layer_call_fn_80807?
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
Q__inference_batch_normalization_53_layer_call_and_return_conditional_losses_80891
Q__inference_batch_normalization_53_layer_call_and_return_conditional_losses_80845
Q__inference_batch_normalization_53_layer_call_and_return_conditional_losses_80909
Q__inference_batch_normalization_53_layer_call_and_return_conditional_losses_80827?
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
6__inference_batch_normalization_53_layer_call_fn_80871
6__inference_batch_normalization_53_layer_call_fn_80922
6__inference_batch_normalization_53_layer_call_fn_80935
6__inference_batch_normalization_53_layer_call_fn_80858?
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
H__inference_activation_53_layer_call_and_return_conditional_losses_80940?
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
-__inference_activation_53_layer_call_fn_80945?
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
N__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_77113?
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
3__inference_conv2d_transpose_11_layer_call_fn_77123?
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
I__inference_concatenate_11_layer_call_and_return_conditional_losses_80952?
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
.__inference_concatenate_11_layer_call_fn_80958?
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
E__inference_dropout_23_layer_call_and_return_conditional_losses_80970
E__inference_dropout_23_layer_call_and_return_conditional_losses_80975?
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
*__inference_dropout_23_layer_call_fn_80985
*__inference_dropout_23_layer_call_fn_80980?
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
D__inference_conv2d_58_layer_call_and_return_conditional_losses_80997?
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
)__inference_conv2d_58_layer_call_fn_81006?
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
Q__inference_batch_normalization_55_layer_call_and_return_conditional_losses_81026
Q__inference_batch_normalization_55_layer_call_and_return_conditional_losses_81090
Q__inference_batch_normalization_55_layer_call_and_return_conditional_losses_81108
Q__inference_batch_normalization_55_layer_call_and_return_conditional_losses_81044?
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
6__inference_batch_normalization_55_layer_call_fn_81070
6__inference_batch_normalization_55_layer_call_fn_81121
6__inference_batch_normalization_55_layer_call_fn_81134
6__inference_batch_normalization_55_layer_call_fn_81057?
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
H__inference_activation_55_layer_call_and_return_conditional_losses_81139?
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
-__inference_activation_55_layer_call_fn_81144?
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
D__inference_conv2d_59_layer_call_and_return_conditional_losses_81157?
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
)__inference_conv2d_59_layer_call_fn_81166?
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
#__inference_signature_wrapper_79058input_4"?
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
 __inference__wrapped_model_76325?P-.4567HIOPQRcdjklm~??????????????????????????????:?7
0?-
+?(
input_4???????????
? "??<
:
	conv2d_59-?*
	conv2d_59????????????
H__inference_activation_43_layer_call_and_return_conditional_losses_79984l9?6
/?,
*?'
inputs??????????? 
? "/?,
%?"
0??????????? 
? ?
-__inference_activation_43_layer_call_fn_79989_9?6
/?,
*?'
inputs??????????? 
? ""???????????? ?
H__inference_activation_45_layer_call_and_return_conditional_losses_80170l9?6
/?,
*?'
inputs???????????@
? "/?,
%?"
0???????????@
? ?
-__inference_activation_45_layer_call_fn_80175_9?6
/?,
*?'
inputs???????????@
? ""????????????@?
H__inference_activation_47_layer_call_and_return_conditional_losses_80356n:?7
0?-
+?(
inputs????????????
? "0?-
&?#
0????????????
? ?
-__inference_activation_47_layer_call_fn_80361a:?7
0?-
+?(
inputs????????????
? "#? ?????????????
H__inference_activation_49_layer_call_and_return_conditional_losses_80542j8?5
.?+
)?&
inputs?????????D\?
? ".?+
$?!
0?????????D\?
? ?
-__inference_activation_49_layer_call_fn_80547]8?5
.?+
)?&
inputs?????????D\?
? "!??????????D\??
H__inference_activation_51_layer_call_and_return_conditional_losses_80741n:?7
0?-
+?(
inputs????????????
? "0?-
&?#
0????????????
? ?
-__inference_activation_51_layer_call_fn_80746a:?7
0?-
+?(
inputs????????????
? "#? ?????????????
H__inference_activation_53_layer_call_and_return_conditional_losses_80940l9?6
/?,
*?'
inputs???????????@
? "/?,
%?"
0???????????@
? ?
-__inference_activation_53_layer_call_fn_80945_9?6
/?,
*?'
inputs???????????@
? ""????????????@?
H__inference_activation_55_layer_call_and_return_conditional_losses_81139l9?6
/?,
*?'
inputs??????????? 
? "/?,
%?"
0??????????? 
? ?
-__inference_activation_55_layer_call_fn_81144_9?6
/?,
*?'
inputs??????????? 
? ""???????????? ?
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_79871?4567M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_79889?4567M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_79935v4567=?:
3?0
*?'
inputs??????????? 
p
? "/?,
%?"
0??????????? 
? ?
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_79953v4567=?:
3?0
*?'
inputs??????????? 
p 
? "/?,
%?"
0??????????? 
? ?
6__inference_batch_normalization_43_layer_call_fn_79902?4567M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
6__inference_batch_normalization_43_layer_call_fn_79915?4567M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
6__inference_batch_normalization_43_layer_call_fn_79966i4567=?:
3?0
*?'
inputs??????????? 
p
? ""???????????? ?
6__inference_batch_normalization_43_layer_call_fn_79979i4567=?:
3?0
*?'
inputs??????????? 
p 
? ""???????????? ?
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_80057vOPQR=?:
3?0
*?'
inputs???????????@
p
? "/?,
%?"
0???????????@
? ?
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_80075vOPQR=?:
3?0
*?'
inputs???????????@
p 
? "/?,
%?"
0???????????@
? ?
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_80121?OPQRM?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_80139?OPQRM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
6__inference_batch_normalization_45_layer_call_fn_80088iOPQR=?:
3?0
*?'
inputs???????????@
p
? ""????????????@?
6__inference_batch_normalization_45_layer_call_fn_80101iOPQR=?:
3?0
*?'
inputs???????????@
p 
? ""????????????@?
6__inference_batch_normalization_45_layer_call_fn_80152?OPQRM?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
6__inference_batch_normalization_45_layer_call_fn_80165?OPQRM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
Q__inference_batch_normalization_47_layer_call_and_return_conditional_losses_80243xjklm>?;
4?1
+?(
inputs????????????
p
? "0?-
&?#
0????????????
? ?
Q__inference_batch_normalization_47_layer_call_and_return_conditional_losses_80261xjklm>?;
4?1
+?(
inputs????????????
p 
? "0?-
&?#
0????????????
? ?
Q__inference_batch_normalization_47_layer_call_and_return_conditional_losses_80307?jklmN?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
Q__inference_batch_normalization_47_layer_call_and_return_conditional_losses_80325?jklmN?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
6__inference_batch_normalization_47_layer_call_fn_80274kjklm>?;
4?1
+?(
inputs????????????
p
? "#? ?????????????
6__inference_batch_normalization_47_layer_call_fn_80287kjklm>?;
4?1
+?(
inputs????????????
p 
? "#? ?????????????
6__inference_batch_normalization_47_layer_call_fn_80338?jklmN?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
6__inference_batch_normalization_47_layer_call_fn_80351?jklmN?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_80429x????<?9
2?/
)?&
inputs?????????D\?
p
? ".?+
$?!
0?????????D\?
? ?
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_80447x????<?9
2?/
)?&
inputs?????????D\?
p 
? ".?+
$?!
0?????????D\?
? ?
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_80493?????N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_80511?????N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
6__inference_batch_normalization_49_layer_call_fn_80460k????<?9
2?/
)?&
inputs?????????D\?
p
? "!??????????D\??
6__inference_batch_normalization_49_layer_call_fn_80473k????<?9
2?/
)?&
inputs?????????D\?
p 
? "!??????????D\??
6__inference_batch_normalization_49_layer_call_fn_80524?????N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
6__inference_batch_normalization_49_layer_call_fn_80537?????N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_80628?????N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_80646?????N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_80692|????>?;
4?1
+?(
inputs????????????
p
? "0?-
&?#
0????????????
? ?
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_80710|????>?;
4?1
+?(
inputs????????????
p 
? "0?-
&?#
0????????????
? ?
6__inference_batch_normalization_51_layer_call_fn_80659?????N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
6__inference_batch_normalization_51_layer_call_fn_80672?????N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
6__inference_batch_normalization_51_layer_call_fn_80723o????>?;
4?1
+?(
inputs????????????
p
? "#? ?????????????
6__inference_batch_normalization_51_layer_call_fn_80736o????>?;
4?1
+?(
inputs????????????
p 
? "#? ?????????????
Q__inference_batch_normalization_53_layer_call_and_return_conditional_losses_80827?????M?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
Q__inference_batch_normalization_53_layer_call_and_return_conditional_losses_80845?????M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
Q__inference_batch_normalization_53_layer_call_and_return_conditional_losses_80891z????=?:
3?0
*?'
inputs???????????@
p
? "/?,
%?"
0???????????@
? ?
Q__inference_batch_normalization_53_layer_call_and_return_conditional_losses_80909z????=?:
3?0
*?'
inputs???????????@
p 
? "/?,
%?"
0???????????@
? ?
6__inference_batch_normalization_53_layer_call_fn_80858?????M?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
6__inference_batch_normalization_53_layer_call_fn_80871?????M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
6__inference_batch_normalization_53_layer_call_fn_80922m????=?:
3?0
*?'
inputs???????????@
p
? ""????????????@?
6__inference_batch_normalization_53_layer_call_fn_80935m????=?:
3?0
*?'
inputs???????????@
p 
? ""????????????@?
Q__inference_batch_normalization_55_layer_call_and_return_conditional_losses_81026z????=?:
3?0
*?'
inputs??????????? 
p
? "/?,
%?"
0??????????? 
? ?
Q__inference_batch_normalization_55_layer_call_and_return_conditional_losses_81044z????=?:
3?0
*?'
inputs??????????? 
p 
? "/?,
%?"
0??????????? 
? ?
Q__inference_batch_normalization_55_layer_call_and_return_conditional_losses_81090?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
Q__inference_batch_normalization_55_layer_call_and_return_conditional_losses_81108?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
6__inference_batch_normalization_55_layer_call_fn_81057m????=?:
3?0
*?'
inputs??????????? 
p
? ""???????????? ?
6__inference_batch_normalization_55_layer_call_fn_81070m????=?:
3?0
*?'
inputs??????????? 
p 
? ""???????????? ?
6__inference_batch_normalization_55_layer_call_fn_81121?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
6__inference_batch_normalization_55_layer_call_fn_81134?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
I__inference_concatenate_10_layer_call_and_return_conditional_losses_80753?~?{
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
.__inference_concatenate_10_layer_call_fn_80759?~?{
t?q
o?l
<?9
inputs/0+???????????????????????????@
,?)
inputs/1???????????@
? "#? ?????????????
I__inference_concatenate_11_layer_call_and_return_conditional_losses_80952?~?{
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
.__inference_concatenate_11_layer_call_fn_80958?~?{
t?q
o?l
<?9
inputs/0+??????????????????????????? 
,?)
inputs/1??????????? 
? ""????????????@?
H__inference_concatenate_9_layer_call_and_return_conditional_losses_80554???}
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
-__inference_concatenate_9_layer_call_fn_80560???}
v?s
q?n
=?:
inputs/0,????????????????????????????
-?*
inputs/1????????????
? "#? ?????????????
D__inference_conv2d_46_layer_call_and_return_conditional_losses_79842p-.9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0??????????? 
? ?
)__inference_conv2d_46_layer_call_fn_79851c-.9?6
/?,
*?'
inputs???????????
? ""???????????? ?
D__inference_conv2d_48_layer_call_and_return_conditional_losses_80028pHI9?6
/?,
*?'
inputs??????????? 
? "/?,
%?"
0???????????@
? ?
)__inference_conv2d_48_layer_call_fn_80037cHI9?6
/?,
*?'
inputs??????????? 
? ""????????????@?
D__inference_conv2d_50_layer_call_and_return_conditional_losses_80214qcd9?6
/?,
*?'
inputs???????????@
? "0?-
&?#
0????????????
? ?
)__inference_conv2d_50_layer_call_fn_80223dcd9?6
/?,
*?'
inputs???????????@
? "#? ?????????????
D__inference_conv2d_52_layer_call_and_return_conditional_losses_80400n~8?5
.?+
)?&
inputs?????????D\?
? ".?+
$?!
0?????????D\?
? ?
)__inference_conv2d_52_layer_call_fn_80409a~8?5
.?+
)?&
inputs?????????D\?
? "!??????????D\??
D__inference_conv2d_54_layer_call_and_return_conditional_losses_80599t??:?7
0?-
+?(
inputs????????????
? "0?-
&?#
0????????????
? ?
)__inference_conv2d_54_layer_call_fn_80608g??:?7
0?-
+?(
inputs????????????
? "#? ?????????????
D__inference_conv2d_56_layer_call_and_return_conditional_losses_80798s??:?7
0?-
+?(
inputs????????????
? "/?,
%?"
0???????????@
? ?
)__inference_conv2d_56_layer_call_fn_80807f??:?7
0?-
+?(
inputs????????????
? ""????????????@?
D__inference_conv2d_58_layer_call_and_return_conditional_losses_80997r??9?6
/?,
*?'
inputs???????????@
? "/?,
%?"
0??????????? 
? ?
)__inference_conv2d_58_layer_call_fn_81006e??9?6
/?,
*?'
inputs???????????@
? ""???????????? ?
D__inference_conv2d_59_layer_call_and_return_conditional_losses_81157r??9?6
/?,
*?'
inputs??????????? 
? "/?,
%?"
0???????????
? ?
)__inference_conv2d_59_layer_call_fn_81166e??9?6
/?,
*?'
inputs??????????? 
? ""?????????????
N__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_76963???J?G
@?=
;?8
inputs,????????????????????????????
? "??<
5?2
0+???????????????????????????@
? ?
3__inference_conv2d_transpose_10_layer_call_fn_76973???J?G
@?=
;?8
inputs,????????????????????????????
? "2?/+???????????????????????????@?
N__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_77113???I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+??????????????????????????? 
? ?
3__inference_conv2d_transpose_11_layer_call_fn_77123???I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+??????????????????????????? ?
M__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_76813???J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
2__inference_conv2d_transpose_9_layer_call_fn_76823???J?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
E__inference_dropout_18_layer_call_and_return_conditional_losses_80001p=?:
3?0
*?'
inputs??????????? 
p
? "/?,
%?"
0??????????? 
? ?
E__inference_dropout_18_layer_call_and_return_conditional_losses_80006p=?:
3?0
*?'
inputs??????????? 
p 
? "/?,
%?"
0??????????? 
? ?
*__inference_dropout_18_layer_call_fn_80011c=?:
3?0
*?'
inputs??????????? 
p
? ""???????????? ?
*__inference_dropout_18_layer_call_fn_80016c=?:
3?0
*?'
inputs??????????? 
p 
? ""???????????? ?
E__inference_dropout_19_layer_call_and_return_conditional_losses_80187p=?:
3?0
*?'
inputs???????????@
p
? "/?,
%?"
0???????????@
? ?
E__inference_dropout_19_layer_call_and_return_conditional_losses_80192p=?:
3?0
*?'
inputs???????????@
p 
? "/?,
%?"
0???????????@
? ?
*__inference_dropout_19_layer_call_fn_80197c=?:
3?0
*?'
inputs???????????@
p
? ""????????????@?
*__inference_dropout_19_layer_call_fn_80202c=?:
3?0
*?'
inputs???????????@
p 
? ""????????????@?
E__inference_dropout_20_layer_call_and_return_conditional_losses_80373n<?9
2?/
)?&
inputs?????????D\?
p
? ".?+
$?!
0?????????D\?
? ?
E__inference_dropout_20_layer_call_and_return_conditional_losses_80378n<?9
2?/
)?&
inputs?????????D\?
p 
? ".?+
$?!
0?????????D\?
? ?
*__inference_dropout_20_layer_call_fn_80383a<?9
2?/
)?&
inputs?????????D\?
p
? "!??????????D\??
*__inference_dropout_20_layer_call_fn_80388a<?9
2?/
)?&
inputs?????????D\?
p 
? "!??????????D\??
E__inference_dropout_21_layer_call_and_return_conditional_losses_80572r>?;
4?1
+?(
inputs????????????
p
? "0?-
&?#
0????????????
? ?
E__inference_dropout_21_layer_call_and_return_conditional_losses_80577r>?;
4?1
+?(
inputs????????????
p 
? "0?-
&?#
0????????????
? ?
*__inference_dropout_21_layer_call_fn_80582e>?;
4?1
+?(
inputs????????????
p
? "#? ?????????????
*__inference_dropout_21_layer_call_fn_80587e>?;
4?1
+?(
inputs????????????
p 
? "#? ?????????????
E__inference_dropout_22_layer_call_and_return_conditional_losses_80771r>?;
4?1
+?(
inputs????????????
p
? "0?-
&?#
0????????????
? ?
E__inference_dropout_22_layer_call_and_return_conditional_losses_80776r>?;
4?1
+?(
inputs????????????
p 
? "0?-
&?#
0????????????
? ?
*__inference_dropout_22_layer_call_fn_80781e>?;
4?1
+?(
inputs????????????
p
? "#? ?????????????
*__inference_dropout_22_layer_call_fn_80786e>?;
4?1
+?(
inputs????????????
p 
? "#? ?????????????
E__inference_dropout_23_layer_call_and_return_conditional_losses_80970p=?:
3?0
*?'
inputs???????????@
p
? "/?,
%?"
0???????????@
? ?
E__inference_dropout_23_layer_call_and_return_conditional_losses_80975p=?:
3?0
*?'
inputs???????????@
p 
? "/?,
%?"
0???????????@
? ?
*__inference_dropout_23_layer_call_fn_80980c=?:
3?0
*?'
inputs???????????@
p
? ""????????????@?
*__inference_dropout_23_layer_call_fn_80985c=?:
3?0
*?'
inputs???????????@
p 
? ""????????????@?
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_76551?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
0__inference_max_pooling2d_10_layer_call_fn_76557?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_76667?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
0__inference_max_pooling2d_11_layer_call_fn_76673?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_76435?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_max_pooling2d_9_layer_call_fn_76441?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
B__inference_model_3_layer_call_and_return_conditional_losses_78306?P-.4567HIOPQRcdjklm~??????????????????????????????B??
8?5
+?(
input_4???????????
p

 
? "/?,
%?"
0???????????
? ?
B__inference_model_3_layer_call_and_return_conditional_losses_78448?P-.4567HIOPQRcdjklm~??????????????????????????????B??
8?5
+?(
input_4???????????
p 

 
? "/?,
%?"
0???????????
? ?
B__inference_model_3_layer_call_and_return_conditional_losses_79367?P-.4567HIOPQRcdjklm~??????????????????????????????A?>
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
B__inference_model_3_layer_call_and_return_conditional_losses_79620?P-.4567HIOPQRcdjklm~??????????????????????????????A?>
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
'__inference_model_3_layer_call_fn_78696?P-.4567HIOPQRcdjklm~??????????????????????????????B??
8?5
+?(
input_4???????????
p

 
? ""?????????????
'__inference_model_3_layer_call_fn_78943?P-.4567HIOPQRcdjklm~??????????????????????????????B??
8?5
+?(
input_4???????????
p 

 
? ""?????????????
'__inference_model_3_layer_call_fn_79725?P-.4567HIOPQRcdjklm~??????????????????????????????A?>
7?4
*?'
inputs???????????
p

 
? ""?????????????
'__inference_model_3_layer_call_fn_79830?P-.4567HIOPQRcdjklm~??????????????????????????????A?>
7?4
*?'
inputs???????????
p 

 
? ""?????????????
#__inference_signature_wrapper_79058?P-.4567HIOPQRcdjklm~??????????????????????????????E?B
? 
;?8
6
input_4+?(
input_4???????????"??<
:
	conv2d_59-?*
	conv2d_59???????????