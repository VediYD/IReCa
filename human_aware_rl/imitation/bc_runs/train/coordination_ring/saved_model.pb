��
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
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
executor_typestring �
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*1.15.02unknown8��
v
fc_0_1/kernelVarHandleOp*
shared_namefc_0_1/kernel*
dtype0*
_output_shapes
: *
shape
:`@
o
!fc_0_1/kernel/Read/ReadVariableOpReadVariableOpfc_0_1/kernel*
_output_shapes

:`@*
dtype0
n
fc_0_1/biasVarHandleOp*
dtype0*
shape:@*
_output_shapes
: *
shared_namefc_0_1/bias
g
fc_0_1/bias/Read/ReadVariableOpReadVariableOpfc_0_1/bias*
_output_shapes
:@*
dtype0
v
fc_1_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*
shared_namefc_1_1/kernel
o
!fc_1_1/kernel/Read/ReadVariableOpReadVariableOpfc_1_1/kernel*
dtype0*
_output_shapes

:@@
n
fc_1_1/biasVarHandleOp*
dtype0*
shape:@*
shared_namefc_1_1/bias*
_output_shapes
: 
g
fc_1_1/bias/Read/ReadVariableOpReadVariableOpfc_1_1/bias*
_output_shapes
:@*
dtype0
z
logits_1/kernelVarHandleOp* 
shared_namelogits_1/kernel*
_output_shapes
: *
dtype0*
shape
:@
s
#logits_1/kernel/Read/ReadVariableOpReadVariableOplogits_1/kernel*
dtype0*
_output_shapes

:@
r
logits_1/biasVarHandleOp*
shape:*
shared_namelogits_1/bias*
_output_shapes
: *
dtype0
k
!logits_1/bias/Read/ReadVariableOpReadVariableOplogits_1/bias*
_output_shapes
:*
dtype0
|
training_2/Adam/iterVarHandleOp*
shape: *%
shared_nametraining_2/Adam/iter*
dtype0	*
_output_shapes
: 
u
(training_2/Adam/iter/Read/ReadVariableOpReadVariableOptraining_2/Adam/iter*
dtype0	*
_output_shapes
: 
�
training_2/Adam/beta_1VarHandleOp*
dtype0*
_output_shapes
: *
shape: *'
shared_nametraining_2/Adam/beta_1
y
*training_2/Adam/beta_1/Read/ReadVariableOpReadVariableOptraining_2/Adam/beta_1*
dtype0*
_output_shapes
: 
�
training_2/Adam/beta_2VarHandleOp*'
shared_nametraining_2/Adam/beta_2*
dtype0*
shape: *
_output_shapes
: 
y
*training_2/Adam/beta_2/Read/ReadVariableOpReadVariableOptraining_2/Adam/beta_2*
_output_shapes
: *
dtype0
~
training_2/Adam/decayVarHandleOp*&
shared_nametraining_2/Adam/decay*
dtype0*
_output_shapes
: *
shape: 
w
)training_2/Adam/decay/Read/ReadVariableOpReadVariableOptraining_2/Adam/decay*
dtype0*
_output_shapes
: 
�
training_2/Adam/learning_rateVarHandleOp*
shape: *.
shared_nametraining_2/Adam/learning_rate*
_output_shapes
: *
dtype0
�
1training_2/Adam/learning_rate/Read/ReadVariableOpReadVariableOptraining_2/Adam/learning_rate*
dtype0*
_output_shapes
: 
b
total_1VarHandleOp*
shape: *
_output_shapes
: *
dtype0*
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
dtype0*
_output_shapes
: 
b
count_1VarHandleOp*
shared_name	count_1*
dtype0*
_output_shapes
: *
shape: 
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
�
training_2/Adam/fc_0_1/kernel/mVarHandleOp*
dtype0*0
shared_name!training_2/Adam/fc_0_1/kernel/m*
_output_shapes
: *
shape
:`@
�
3training_2/Adam/fc_0_1/kernel/m/Read/ReadVariableOpReadVariableOptraining_2/Adam/fc_0_1/kernel/m*
_output_shapes

:`@*
dtype0
�
training_2/Adam/fc_0_1/bias/mVarHandleOp*
shape:@*
_output_shapes
: *.
shared_nametraining_2/Adam/fc_0_1/bias/m*
dtype0
�
1training_2/Adam/fc_0_1/bias/m/Read/ReadVariableOpReadVariableOptraining_2/Adam/fc_0_1/bias/m*
dtype0*
_output_shapes
:@
�
training_2/Adam/fc_1_1/kernel/mVarHandleOp*
dtype0*
_output_shapes
: *0
shared_name!training_2/Adam/fc_1_1/kernel/m*
shape
:@@
�
3training_2/Adam/fc_1_1/kernel/m/Read/ReadVariableOpReadVariableOptraining_2/Adam/fc_1_1/kernel/m*
_output_shapes

:@@*
dtype0
�
training_2/Adam/fc_1_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*.
shared_nametraining_2/Adam/fc_1_1/bias/m*
shape:@
�
1training_2/Adam/fc_1_1/bias/m/Read/ReadVariableOpReadVariableOptraining_2/Adam/fc_1_1/bias/m*
dtype0*
_output_shapes
:@
�
!training_2/Adam/logits_1/kernel/mVarHandleOp*2
shared_name#!training_2/Adam/logits_1/kernel/m*
dtype0*
shape
:@*
_output_shapes
: 
�
5training_2/Adam/logits_1/kernel/m/Read/ReadVariableOpReadVariableOp!training_2/Adam/logits_1/kernel/m*
_output_shapes

:@*
dtype0
�
training_2/Adam/logits_1/bias/mVarHandleOp*0
shared_name!training_2/Adam/logits_1/bias/m*
_output_shapes
: *
shape:*
dtype0
�
3training_2/Adam/logits_1/bias/m/Read/ReadVariableOpReadVariableOptraining_2/Adam/logits_1/bias/m*
dtype0*
_output_shapes
:
�
training_2/Adam/fc_0_1/kernel/vVarHandleOp*
_output_shapes
: *0
shared_name!training_2/Adam/fc_0_1/kernel/v*
shape
:`@*
dtype0
�
3training_2/Adam/fc_0_1/kernel/v/Read/ReadVariableOpReadVariableOptraining_2/Adam/fc_0_1/kernel/v*
dtype0*
_output_shapes

:`@
�
training_2/Adam/fc_0_1/bias/vVarHandleOp*
shape:@*
_output_shapes
: *.
shared_nametraining_2/Adam/fc_0_1/bias/v*
dtype0
�
1training_2/Adam/fc_0_1/bias/v/Read/ReadVariableOpReadVariableOptraining_2/Adam/fc_0_1/bias/v*
dtype0*
_output_shapes
:@
�
training_2/Adam/fc_1_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*0
shared_name!training_2/Adam/fc_1_1/kernel/v
�
3training_2/Adam/fc_1_1/kernel/v/Read/ReadVariableOpReadVariableOptraining_2/Adam/fc_1_1/kernel/v*
_output_shapes

:@@*
dtype0
�
training_2/Adam/fc_1_1/bias/vVarHandleOp*
_output_shapes
: *
shape:@*
dtype0*.
shared_nametraining_2/Adam/fc_1_1/bias/v
�
1training_2/Adam/fc_1_1/bias/v/Read/ReadVariableOpReadVariableOptraining_2/Adam/fc_1_1/bias/v*
dtype0*
_output_shapes
:@
�
!training_2/Adam/logits_1/kernel/vVarHandleOp*
dtype0*
_output_shapes
: *
shape
:@*2
shared_name#!training_2/Adam/logits_1/kernel/v
�
5training_2/Adam/logits_1/kernel/v/Read/ReadVariableOpReadVariableOp!training_2/Adam/logits_1/kernel/v*
_output_shapes

:@*
dtype0
�
training_2/Adam/logits_1/bias/vVarHandleOp*0
shared_name!training_2/Adam/logits_1/bias/v*
shape:*
dtype0*
_output_shapes
: 
�
3training_2/Adam/logits_1/bias/v/Read/ReadVariableOpReadVariableOptraining_2/Adam/logits_1/bias/v*
dtype0*
_output_shapes
:

NoOpNoOp
�(
ConstConst"/device:CPU:0*
_output_shapes
: *�(
value�'B�' B�'
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api


signatures
R
	variables
trainable_variables
regularization_losses
	keras_api
~

kernel
bias
_callable_losses
	variables
trainable_variables
regularization_losses
	keras_api
~

kernel
bias
_callable_losses
	variables
trainable_variables
regularization_losses
	keras_api
~

kernel
bias
_callable_losses
 	variables
!trainable_variables
"regularization_losses
#	keras_api
�
$iter

%beta_1

&beta_2
	'decay
(learning_ratemJmKmLmMmNmOvPvQvRvSvTvU
*
0
1
2
3
4
5
*
0
1
2
3
4
5
 
�
)layer_regularization_losses
	variables
*metrics
trainable_variables
regularization_losses

+layers
,non_trainable_variables
 
 
 
 
�
-layer_regularization_losses
	variables
.metrics
trainable_variables
regularization_losses

/layers
0non_trainable_variables
YW
VARIABLE_VALUEfc_0_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEfc_0_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
 
�
1layer_regularization_losses
	variables
2metrics
trainable_variables
regularization_losses

3layers
4non_trainable_variables
YW
VARIABLE_VALUEfc_1_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEfc_1_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
 
�
5layer_regularization_losses
	variables
6metrics
trainable_variables
regularization_losses

7layers
8non_trainable_variables
[Y
VARIABLE_VALUElogits_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUElogits_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
 
�
9layer_regularization_losses
 	variables
:metrics
!trainable_variables
"regularization_losses

;layers
<non_trainable_variables
SQ
VARIABLE_VALUEtraining_2/Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEtraining_2/Adam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEtraining_2/Adam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEtraining_2/Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEtraining_2/Adam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 

=0

0
1
2
3
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
�
	>total
	?count
@
_fn_kwargs
A_updates
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
QO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 

>0
?1
 
 
�
Flayer_regularization_losses
B	variables
Gmetrics
Ctrainable_variables
Dregularization_losses

Hlayers
Inon_trainable_variables
 
 
 

>0
?1
��
VARIABLE_VALUEtraining_2/Adam/fc_0_1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEtraining_2/Adam/fc_0_1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEtraining_2/Adam/fc_1_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEtraining_2/Adam/fc_1_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE!training_2/Adam/logits_1/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEtraining_2/Adam/logits_1/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEtraining_2/Adam/fc_0_1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEtraining_2/Adam/fc_0_1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEtraining_2/Adam/fc_1_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEtraining_2/Adam/fc_1_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE!training_2/Adam/logits_1/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEtraining_2/Adam/logits_1/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0
�
&serving_default_Overcooked_observationPlaceholder*
dtype0*'
_output_shapes
:���������`*
shape:���������`
�
StatefulPartitionedCallStatefulPartitionedCall&serving_default_Overcooked_observationfc_0_1/kernelfc_0_1/biasfc_1_1/kernelfc_1_1/biaslogits_1/kernellogits_1/bias**
config_proto

CPU

GPU 2J 8*
Tin
	2*'
_output_shapes
:���������*+
_gradient_op_typePartitionedCall-1523*
Tout
2*+
f&R$
"__inference_signature_wrapper_1343
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!fc_0_1/kernel/Read/ReadVariableOpfc_0_1/bias/Read/ReadVariableOp!fc_1_1/kernel/Read/ReadVariableOpfc_1_1/bias/Read/ReadVariableOp#logits_1/kernel/Read/ReadVariableOp!logits_1/bias/Read/ReadVariableOp(training_2/Adam/iter/Read/ReadVariableOp*training_2/Adam/beta_1/Read/ReadVariableOp*training_2/Adam/beta_2/Read/ReadVariableOp)training_2/Adam/decay/Read/ReadVariableOp1training_2/Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp3training_2/Adam/fc_0_1/kernel/m/Read/ReadVariableOp1training_2/Adam/fc_0_1/bias/m/Read/ReadVariableOp3training_2/Adam/fc_1_1/kernel/m/Read/ReadVariableOp1training_2/Adam/fc_1_1/bias/m/Read/ReadVariableOp5training_2/Adam/logits_1/kernel/m/Read/ReadVariableOp3training_2/Adam/logits_1/bias/m/Read/ReadVariableOp3training_2/Adam/fc_0_1/kernel/v/Read/ReadVariableOp1training_2/Adam/fc_0_1/bias/v/Read/ReadVariableOp3training_2/Adam/fc_1_1/kernel/v/Read/ReadVariableOp1training_2/Adam/fc_1_1/bias/v/Read/ReadVariableOp5training_2/Adam/logits_1/kernel/v/Read/ReadVariableOp3training_2/Adam/logits_1/bias/v/Read/ReadVariableOpConst*&
Tin
2	*
Tout
2*
_output_shapes
: **
config_proto

CPU

GPU 2J 8*+
_gradient_op_typePartitionedCall-1570*&
f!R
__inference__traced_save_1569
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamefc_0_1/kernelfc_0_1/biasfc_1_1/kernelfc_1_1/biaslogits_1/kernellogits_1/biastraining_2/Adam/itertraining_2/Adam/beta_1training_2/Adam/beta_2training_2/Adam/decaytraining_2/Adam/learning_ratetotal_1count_1training_2/Adam/fc_0_1/kernel/mtraining_2/Adam/fc_0_1/bias/mtraining_2/Adam/fc_1_1/kernel/mtraining_2/Adam/fc_1_1/bias/m!training_2/Adam/logits_1/kernel/mtraining_2/Adam/logits_1/bias/mtraining_2/Adam/fc_0_1/kernel/vtraining_2/Adam/fc_0_1/bias/vtraining_2/Adam/fc_1_1/kernel/vtraining_2/Adam/fc_1_1/bias/v!training_2/Adam/logits_1/kernel/vtraining_2/Adam/logits_1/bias/v*
Tout
2*)
f$R"
 __inference__traced_restore_1657*+
_gradient_op_typePartitionedCall-1658*
_output_shapes
: *%
Tin
2**
config_proto

CPU

GPU 2J 8ͯ
�
�
@__inference_logits_layer_call_and_return_conditional_losses_1462

inputs)
%matmul_readvariableop_logits_1_kernel(
$biasadd_readvariableop_logits_1_bias
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp{
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_logits_1_kernel*
dtype0*
_output_shapes

:@i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������w
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_logits_1_bias*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*.
_input_shapes
:���������@::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
�
�
>__inference_fc_1_layer_call_and_return_conditional_losses_1211

inputs'
#matmul_readvariableop_fc_1_1_kernel&
"biasadd_readvariableop_fc_1_1_bias
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpy
MatMul/ReadVariableOpReadVariableOp#matmul_readvariableop_fc_1_1_kernel*
dtype0*
_output_shapes

:@@i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@u
BiasAdd/ReadVariableOpReadVariableOp"biasadd_readvariableop_fc_1_1_bias*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:���������@*
T0P
ReluReluBiasAdd:output:0*'
_output_shapes
:���������@*
T0�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:���������@*
T0"
identityIdentity:output:0*.
_input_shapes
:���������@::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
�	
�
&__inference_model_1_layer_call_fn_1416

inputs)
%statefulpartitionedcall_fc_0_1_kernel'
#statefulpartitionedcall_fc_0_1_bias)
%statefulpartitionedcall_fc_1_1_kernel'
#statefulpartitionedcall_fc_1_1_bias+
'statefulpartitionedcall_logits_1_kernel)
%statefulpartitionedcall_logits_1_bias
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs%statefulpartitionedcall_fc_0_1_kernel#statefulpartitionedcall_fc_0_1_bias%statefulpartitionedcall_fc_1_1_kernel#statefulpartitionedcall_fc_1_1_bias'statefulpartitionedcall_logits_1_kernel%statefulpartitionedcall_logits_1_bias*
Tout
2*J
fERC
A__inference_model_1_layer_call_and_return_conditional_losses_1320*
Tin
	2*'
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*+
_gradient_op_typePartitionedCall-1321�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*>
_input_shapes-
+:���������`::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : 
�
�
A__inference_model_1_layer_call_and_return_conditional_losses_1320

inputs.
*fc_0_statefulpartitionedcall_fc_0_1_kernel,
(fc_0_statefulpartitionedcall_fc_0_1_bias.
*fc_1_statefulpartitionedcall_fc_1_1_kernel,
(fc_1_statefulpartitionedcall_fc_1_1_bias2
.logits_statefulpartitionedcall_logits_1_kernel0
,logits_statefulpartitionedcall_logits_1_bias
identity��fc_0/StatefulPartitionedCall�fc_1/StatefulPartitionedCall�logits/StatefulPartitionedCall�
fc_0/StatefulPartitionedCallStatefulPartitionedCallinputs*fc_0_statefulpartitionedcall_fc_0_1_kernel(fc_0_statefulpartitionedcall_fc_0_1_bias*+
_gradient_op_typePartitionedCall-1188*'
_output_shapes
:���������@*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*G
fBR@
>__inference_fc_0_layer_call_and_return_conditional_losses_1181�
fc_1/StatefulPartitionedCallStatefulPartitionedCall%fc_0/StatefulPartitionedCall:output:0*fc_1_statefulpartitionedcall_fc_1_1_kernel(fc_1_statefulpartitionedcall_fc_1_1_bias*
Tin
2*+
_gradient_op_typePartitionedCall-1218*'
_output_shapes
:���������@**
config_proto

CPU

GPU 2J 8*G
fBR@
>__inference_fc_1_layer_call_and_return_conditional_losses_1211*
Tout
2�
logits/StatefulPartitionedCallStatefulPartitionedCall%fc_1/StatefulPartitionedCall:output:0.logits_statefulpartitionedcall_logits_1_kernel,logits_statefulpartitionedcall_logits_1_bias*+
_gradient_op_typePartitionedCall-1247*I
fDRB
@__inference_logits_layer_call_and_return_conditional_losses_1240**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:���������*
Tout
2*
Tin
2�
IdentityIdentity'logits/StatefulPartitionedCall:output:0^fc_0/StatefulPartitionedCall^fc_1/StatefulPartitionedCall^logits/StatefulPartitionedCall*'
_output_shapes
:���������*
T0"
identityIdentity:output:0*>
_input_shapes-
+:���������`::::::2<
fc_0/StatefulPartitionedCallfc_0/StatefulPartitionedCall2<
fc_1/StatefulPartitionedCallfc_1/StatefulPartitionedCall2@
logits/StatefulPartitionedCalllogits/StatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : : : : 
�

�
&__inference_model_1_layer_call_fn_1302
overcooked_observation)
%statefulpartitionedcall_fc_0_1_kernel'
#statefulpartitionedcall_fc_0_1_bias)
%statefulpartitionedcall_fc_1_1_kernel'
#statefulpartitionedcall_fc_1_1_bias+
'statefulpartitionedcall_logits_1_kernel)
%statefulpartitionedcall_logits_1_bias
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallovercooked_observation%statefulpartitionedcall_fc_0_1_kernel#statefulpartitionedcall_fc_0_1_bias%statefulpartitionedcall_fc_1_1_kernel#statefulpartitionedcall_fc_1_1_bias'statefulpartitionedcall_logits_1_kernel%statefulpartitionedcall_logits_1_bias**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:���������*
Tout
2*
Tin
	2*+
_gradient_op_typePartitionedCall-1293*J
fERC
A__inference_model_1_layer_call_and_return_conditional_losses_1292�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:���������*
T0"
identityIdentity:output:0*>
_input_shapes-
+:���������`::::::22
StatefulPartitionedCallStatefulPartitionedCall:6 2
0
_user_specified_nameOvercooked_observation: : : : : : 
�d
�
 __inference__traced_restore_1657
file_prefix"
assignvariableop_fc_0_1_kernel"
assignvariableop_1_fc_0_1_bias$
 assignvariableop_2_fc_1_1_kernel"
assignvariableop_3_fc_1_1_bias&
"assignvariableop_4_logits_1_kernel$
 assignvariableop_5_logits_1_bias+
'assignvariableop_6_training_2_adam_iter-
)assignvariableop_7_training_2_adam_beta_1-
)assignvariableop_8_training_2_adam_beta_2,
(assignvariableop_9_training_2_adam_decay5
1assignvariableop_10_training_2_adam_learning_rate
assignvariableop_11_total_1
assignvariableop_12_count_17
3assignvariableop_13_training_2_adam_fc_0_1_kernel_m5
1assignvariableop_14_training_2_adam_fc_0_1_bias_m7
3assignvariableop_15_training_2_adam_fc_1_1_kernel_m5
1assignvariableop_16_training_2_adam_fc_1_1_bias_m9
5assignvariableop_17_training_2_adam_logits_1_kernel_m7
3assignvariableop_18_training_2_adam_logits_1_bias_m7
3assignvariableop_19_training_2_adam_fc_0_1_kernel_v5
1assignvariableop_20_training_2_adam_fc_0_1_bias_v7
3assignvariableop_21_training_2_adam_fc_1_1_kernel_v5
1assignvariableop_22_training_2_adam_fc_1_1_bias_v9
5assignvariableop_23_training_2_adam_logits_1_kernel_v7
3assignvariableop_24_training_2_adam_logits_1_bias_v
identity_26��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	RestoreV2�RestoreV2_1�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*x
_output_shapesf
d:::::::::::::::::::::::::*'
dtypes
2	L
IdentityIdentityRestoreV2:tensors:0*
_output_shapes
:*
T0z
AssignVariableOpAssignVariableOpassignvariableop_fc_0_1_kernelIdentity:output:0*
_output_shapes
 *
dtype0N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:~
AssignVariableOp_1AssignVariableOpassignvariableop_1_fc_0_1_biasIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp assignvariableop_2_fc_1_1_kernelIdentity_2:output:0*
_output_shapes
 *
dtype0N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:~
AssignVariableOp_3AssignVariableOpassignvariableop_3_fc_1_1_biasIdentity_3:output:0*
_output_shapes
 *
dtype0N

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_logits_1_kernelIdentity_4:output:0*
_output_shapes
 *
dtype0N

Identity_5IdentityRestoreV2:tensors:5*
_output_shapes
:*
T0�
AssignVariableOp_5AssignVariableOp assignvariableop_5_logits_1_biasIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
_output_shapes
:*
T0	�
AssignVariableOp_6AssignVariableOp'assignvariableop_6_training_2_adam_iterIdentity_6:output:0*
dtype0	*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
_output_shapes
:*
T0�
AssignVariableOp_7AssignVariableOp)assignvariableop_7_training_2_adam_beta_1Identity_7:output:0*
dtype0*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
_output_shapes
:*
T0�
AssignVariableOp_8AssignVariableOp)assignvariableop_8_training_2_adam_beta_2Identity_8:output:0*
dtype0*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp(assignvariableop_9_training_2_adam_decayIdentity_9:output:0*
_output_shapes
 *
dtype0P
Identity_10IdentityRestoreV2:tensors:10*
_output_shapes
:*
T0�
AssignVariableOp_10AssignVariableOp1assignvariableop_10_training_2_adam_learning_rateIdentity_10:output:0*
_output_shapes
 *
dtype0P
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:}
AssignVariableOp_11AssignVariableOpassignvariableop_11_total_1Identity_11:output:0*
_output_shapes
 *
dtype0P
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:}
AssignVariableOp_12AssignVariableOpassignvariableop_12_count_1Identity_12:output:0*
_output_shapes
 *
dtype0P
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp3assignvariableop_13_training_2_adam_fc_0_1_kernel_mIdentity_13:output:0*
_output_shapes
 *
dtype0P
Identity_14IdentityRestoreV2:tensors:14*
_output_shapes
:*
T0�
AssignVariableOp_14AssignVariableOp1assignvariableop_14_training_2_adam_fc_0_1_bias_mIdentity_14:output:0*
_output_shapes
 *
dtype0P
Identity_15IdentityRestoreV2:tensors:15*
_output_shapes
:*
T0�
AssignVariableOp_15AssignVariableOp3assignvariableop_15_training_2_adam_fc_1_1_kernel_mIdentity_15:output:0*
dtype0*
_output_shapes
 P
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp1assignvariableop_16_training_2_adam_fc_1_1_bias_mIdentity_16:output:0*
_output_shapes
 *
dtype0P
Identity_17IdentityRestoreV2:tensors:17*
_output_shapes
:*
T0�
AssignVariableOp_17AssignVariableOp5assignvariableop_17_training_2_adam_logits_1_kernel_mIdentity_17:output:0*
dtype0*
_output_shapes
 P
Identity_18IdentityRestoreV2:tensors:18*
_output_shapes
:*
T0�
AssignVariableOp_18AssignVariableOp3assignvariableop_18_training_2_adam_logits_1_bias_mIdentity_18:output:0*
_output_shapes
 *
dtype0P
Identity_19IdentityRestoreV2:tensors:19*
_output_shapes
:*
T0�
AssignVariableOp_19AssignVariableOp3assignvariableop_19_training_2_adam_fc_0_1_kernel_vIdentity_19:output:0*
_output_shapes
 *
dtype0P
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp1assignvariableop_20_training_2_adam_fc_0_1_bias_vIdentity_20:output:0*
dtype0*
_output_shapes
 P
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp3assignvariableop_21_training_2_adam_fc_1_1_kernel_vIdentity_21:output:0*
dtype0*
_output_shapes
 P
Identity_22IdentityRestoreV2:tensors:22*
_output_shapes
:*
T0�
AssignVariableOp_22AssignVariableOp1assignvariableop_22_training_2_adam_fc_1_1_bias_vIdentity_22:output:0*
dtype0*
_output_shapes
 P
Identity_23IdentityRestoreV2:tensors:23*
_output_shapes
:*
T0�
AssignVariableOp_23AssignVariableOp5assignvariableop_23_training_2_adam_logits_1_kernel_vIdentity_23:output:0*
dtype0*
_output_shapes
 P
Identity_24IdentityRestoreV2:tensors:24*
_output_shapes
:*
T0�
AssignVariableOp_24AssignVariableOp3assignvariableop_24_training_2_adam_logits_1_bias_vIdentity_24:output:0*
_output_shapes
 *
dtype0�
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
dtypes
2*
_output_shapes
:1
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_25Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: �
Identity_26IdentityIdentity_25:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: "#
identity_26Identity_26:output:0*y
_input_shapesh
f: :::::::::::::::::::::::::2
	RestoreV2	RestoreV22*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112
RestoreV2_1RestoreV2_12*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_19AssignVariableOp_192$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : : : : : 
�
�
A__inference_model_1_layer_call_and_return_conditional_losses_1292

inputs.
*fc_0_statefulpartitionedcall_fc_0_1_kernel,
(fc_0_statefulpartitionedcall_fc_0_1_bias.
*fc_1_statefulpartitionedcall_fc_1_1_kernel,
(fc_1_statefulpartitionedcall_fc_1_1_bias2
.logits_statefulpartitionedcall_logits_1_kernel0
,logits_statefulpartitionedcall_logits_1_bias
identity��fc_0/StatefulPartitionedCall�fc_1/StatefulPartitionedCall�logits/StatefulPartitionedCall�
fc_0/StatefulPartitionedCallStatefulPartitionedCallinputs*fc_0_statefulpartitionedcall_fc_0_1_kernel(fc_0_statefulpartitionedcall_fc_0_1_bias*'
_output_shapes
:���������@*
Tout
2**
config_proto

CPU

GPU 2J 8*+
_gradient_op_typePartitionedCall-1188*G
fBR@
>__inference_fc_0_layer_call_and_return_conditional_losses_1181*
Tin
2�
fc_1/StatefulPartitionedCallStatefulPartitionedCall%fc_0/StatefulPartitionedCall:output:0*fc_1_statefulpartitionedcall_fc_1_1_kernel(fc_1_statefulpartitionedcall_fc_1_1_bias*
Tin
2**
config_proto

CPU

GPU 2J 8*+
_gradient_op_typePartitionedCall-1218*G
fBR@
>__inference_fc_1_layer_call_and_return_conditional_losses_1211*
Tout
2*'
_output_shapes
:���������@�
logits/StatefulPartitionedCallStatefulPartitionedCall%fc_1/StatefulPartitionedCall:output:0.logits_statefulpartitionedcall_logits_1_kernel,logits_statefulpartitionedcall_logits_1_bias*I
fDRB
@__inference_logits_layer_call_and_return_conditional_losses_1240*'
_output_shapes
:���������*
Tin
2*
Tout
2**
config_proto

CPU

GPU 2J 8*+
_gradient_op_typePartitionedCall-1247�
IdentityIdentity'logits/StatefulPartitionedCall:output:0^fc_0/StatefulPartitionedCall^fc_1/StatefulPartitionedCall^logits/StatefulPartitionedCall*'
_output_shapes
:���������*
T0"
identityIdentity:output:0*>
_input_shapes-
+:���������`::::::2@
logits/StatefulPartitionedCalllogits/StatefulPartitionedCall2<
fc_0/StatefulPartitionedCallfc_0/StatefulPartitionedCall2<
fc_1/StatefulPartitionedCallfc_1/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : 
�
�
>__inference_fc_0_layer_call_and_return_conditional_losses_1427

inputs'
#matmul_readvariableop_fc_0_1_kernel&
"biasadd_readvariableop_fc_0_1_bias
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpy
MatMul/ReadVariableOpReadVariableOp#matmul_readvariableop_fc_0_1_kernel*
dtype0*
_output_shapes

:`@i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:���������@*
T0u
BiasAdd/ReadVariableOpReadVariableOp"biasadd_readvariableop_fc_0_1_bias*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:���������@*
T0"
identityIdentity:output:0*.
_input_shapes
:���������`::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
�
�
A__inference_model_1_layer_call_and_return_conditional_losses_1260
overcooked_observation.
*fc_0_statefulpartitionedcall_fc_0_1_kernel,
(fc_0_statefulpartitionedcall_fc_0_1_bias.
*fc_1_statefulpartitionedcall_fc_1_1_kernel,
(fc_1_statefulpartitionedcall_fc_1_1_bias2
.logits_statefulpartitionedcall_logits_1_kernel0
,logits_statefulpartitionedcall_logits_1_bias
identity��fc_0/StatefulPartitionedCall�fc_1/StatefulPartitionedCall�logits/StatefulPartitionedCall�
fc_0/StatefulPartitionedCallStatefulPartitionedCallovercooked_observation*fc_0_statefulpartitionedcall_fc_0_1_kernel(fc_0_statefulpartitionedcall_fc_0_1_bias**
config_proto

CPU

GPU 2J 8*G
fBR@
>__inference_fc_0_layer_call_and_return_conditional_losses_1181*'
_output_shapes
:���������@*+
_gradient_op_typePartitionedCall-1188*
Tin
2*
Tout
2�
fc_1/StatefulPartitionedCallStatefulPartitionedCall%fc_0/StatefulPartitionedCall:output:0*fc_1_statefulpartitionedcall_fc_1_1_kernel(fc_1_statefulpartitionedcall_fc_1_1_bias**
config_proto

CPU

GPU 2J 8*G
fBR@
>__inference_fc_1_layer_call_and_return_conditional_losses_1211*'
_output_shapes
:���������@*+
_gradient_op_typePartitionedCall-1218*
Tout
2*
Tin
2�
logits/StatefulPartitionedCallStatefulPartitionedCall%fc_1/StatefulPartitionedCall:output:0.logits_statefulpartitionedcall_logits_1_kernel,logits_statefulpartitionedcall_logits_1_bias*'
_output_shapes
:���������*
Tout
2**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_logits_layer_call_and_return_conditional_losses_1240*+
_gradient_op_typePartitionedCall-1247*
Tin
2�
IdentityIdentity'logits/StatefulPartitionedCall:output:0^fc_0/StatefulPartitionedCall^fc_1/StatefulPartitionedCall^logits/StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*>
_input_shapes-
+:���������`::::::2<
fc_0/StatefulPartitionedCallfc_0/StatefulPartitionedCall2<
fc_1/StatefulPartitionedCallfc_1/StatefulPartitionedCall2@
logits/StatefulPartitionedCalllogits/StatefulPartitionedCall:6 2
0
_user_specified_nameOvercooked_observation: : : : : : 
�
�
%__inference_logits_layer_call_fn_1469

inputs+
'statefulpartitionedcall_logits_1_kernel)
%statefulpartitionedcall_logits_1_bias
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs'statefulpartitionedcall_logits_1_kernel%statefulpartitionedcall_logits_1_bias*I
fDRB
@__inference_logits_layer_call_and_return_conditional_losses_1240*
Tin
2*+
_gradient_op_typePartitionedCall-1247*'
_output_shapes
:���������*
Tout
2**
config_proto

CPU

GPU 2J 8�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:���������*
T0"
identityIdentity:output:0*.
_input_shapes
:���������@::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
�
�
A__inference_model_1_layer_call_and_return_conditional_losses_1370

inputs,
(fc_0_matmul_readvariableop_fc_0_1_kernel+
'fc_0_biasadd_readvariableop_fc_0_1_bias,
(fc_1_matmul_readvariableop_fc_1_1_kernel+
'fc_1_biasadd_readvariableop_fc_1_1_bias0
,logits_matmul_readvariableop_logits_1_kernel/
+logits_biasadd_readvariableop_logits_1_bias
identity��fc_0/BiasAdd/ReadVariableOp�fc_0/MatMul/ReadVariableOp�fc_1/BiasAdd/ReadVariableOp�fc_1/MatMul/ReadVariableOp�logits/BiasAdd/ReadVariableOp�logits/MatMul/ReadVariableOp�
fc_0/MatMul/ReadVariableOpReadVariableOp(fc_0_matmul_readvariableop_fc_0_1_kernel*
_output_shapes

:`@*
dtype0s
fc_0/MatMulMatMulinputs"fc_0/MatMul/ReadVariableOp:value:0*'
_output_shapes
:���������@*
T0
fc_0/BiasAdd/ReadVariableOpReadVariableOp'fc_0_biasadd_readvariableop_fc_0_1_bias*
_output_shapes
:@*
dtype0�
fc_0/BiasAddBiasAddfc_0/MatMul:product:0#fc_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@Z
	fc_0/ReluRelufc_0/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
fc_1/MatMul/ReadVariableOpReadVariableOp(fc_1_matmul_readvariableop_fc_1_1_kernel*
_output_shapes

:@@*
dtype0�
fc_1/MatMulMatMulfc_0/Relu:activations:0"fc_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@
fc_1/BiasAdd/ReadVariableOpReadVariableOp'fc_1_biasadd_readvariableop_fc_1_1_bias*
_output_shapes
:@*
dtype0�
fc_1/BiasAddBiasAddfc_1/MatMul:product:0#fc_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@Z
	fc_1/ReluRelufc_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
logits/MatMul/ReadVariableOpReadVariableOp,logits_matmul_readvariableop_logits_1_kernel*
dtype0*
_output_shapes

:@�
logits/MatMulMatMulfc_1/Relu:activations:0$logits/MatMul/ReadVariableOp:value:0*'
_output_shapes
:���������*
T0�
logits/BiasAdd/ReadVariableOpReadVariableOp+logits_biasadd_readvariableop_logits_1_bias*
_output_shapes
:*
dtype0�
logits/BiasAddBiasAddlogits/MatMul:product:0%logits/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:���������*
T0�
IdentityIdentitylogits/BiasAdd:output:0^fc_0/BiasAdd/ReadVariableOp^fc_0/MatMul/ReadVariableOp^fc_1/BiasAdd/ReadVariableOp^fc_1/MatMul/ReadVariableOp^logits/BiasAdd/ReadVariableOp^logits/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*>
_input_shapes-
+:���������`::::::28
fc_0/MatMul/ReadVariableOpfc_0/MatMul/ReadVariableOp2:
fc_1/BiasAdd/ReadVariableOpfc_1/BiasAdd/ReadVariableOp2:
fc_0/BiasAdd/ReadVariableOpfc_0/BiasAdd/ReadVariableOp2>
logits/BiasAdd/ReadVariableOplogits/BiasAdd/ReadVariableOp28
fc_1/MatMul/ReadVariableOpfc_1/MatMul/ReadVariableOp2<
logits/MatMul/ReadVariableOplogits/MatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: : : : : 
�
�
__inference__wrapped_model_1163
overcooked_observation4
0model_1_fc_0_matmul_readvariableop_fc_0_1_kernel3
/model_1_fc_0_biasadd_readvariableop_fc_0_1_bias4
0model_1_fc_1_matmul_readvariableop_fc_1_1_kernel3
/model_1_fc_1_biasadd_readvariableop_fc_1_1_bias8
4model_1_logits_matmul_readvariableop_logits_1_kernel7
3model_1_logits_biasadd_readvariableop_logits_1_bias
identity��#model_1/fc_0/BiasAdd/ReadVariableOp�"model_1/fc_0/MatMul/ReadVariableOp�#model_1/fc_1/BiasAdd/ReadVariableOp�"model_1/fc_1/MatMul/ReadVariableOp�%model_1/logits/BiasAdd/ReadVariableOp�$model_1/logits/MatMul/ReadVariableOp�
"model_1/fc_0/MatMul/ReadVariableOpReadVariableOp0model_1_fc_0_matmul_readvariableop_fc_0_1_kernel*
_output_shapes

:`@*
dtype0�
model_1/fc_0/MatMulMatMulovercooked_observation*model_1/fc_0/MatMul/ReadVariableOp:value:0*'
_output_shapes
:���������@*
T0�
#model_1/fc_0/BiasAdd/ReadVariableOpReadVariableOp/model_1_fc_0_biasadd_readvariableop_fc_0_1_bias*
_output_shapes
:@*
dtype0�
model_1/fc_0/BiasAddBiasAddmodel_1/fc_0/MatMul:product:0+model_1/fc_0/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:���������@*
T0j
model_1/fc_0/ReluRelumodel_1/fc_0/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
"model_1/fc_1/MatMul/ReadVariableOpReadVariableOp0model_1_fc_1_matmul_readvariableop_fc_1_1_kernel*
dtype0*
_output_shapes

:@@�
model_1/fc_1/MatMulMatMulmodel_1/fc_0/Relu:activations:0*model_1/fc_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
#model_1/fc_1/BiasAdd/ReadVariableOpReadVariableOp/model_1_fc_1_biasadd_readvariableop_fc_1_1_bias*
dtype0*
_output_shapes
:@�
model_1/fc_1/BiasAddBiasAddmodel_1/fc_1/MatMul:product:0+model_1/fc_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@j
model_1/fc_1/ReluRelumodel_1/fc_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
$model_1/logits/MatMul/ReadVariableOpReadVariableOp4model_1_logits_matmul_readvariableop_logits_1_kernel*
_output_shapes

:@*
dtype0�
model_1/logits/MatMulMatMulmodel_1/fc_1/Relu:activations:0,model_1/logits/MatMul/ReadVariableOp:value:0*'
_output_shapes
:���������*
T0�
%model_1/logits/BiasAdd/ReadVariableOpReadVariableOp3model_1_logits_biasadd_readvariableop_logits_1_bias*
_output_shapes
:*
dtype0�
model_1/logits/BiasAddBiasAddmodel_1/logits/MatMul:product:0-model_1/logits/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:���������*
T0�
IdentityIdentitymodel_1/logits/BiasAdd:output:0$^model_1/fc_0/BiasAdd/ReadVariableOp#^model_1/fc_0/MatMul/ReadVariableOp$^model_1/fc_1/BiasAdd/ReadVariableOp#^model_1/fc_1/MatMul/ReadVariableOp&^model_1/logits/BiasAdd/ReadVariableOp%^model_1/logits/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*>
_input_shapes-
+:���������`::::::2J
#model_1/fc_0/BiasAdd/ReadVariableOp#model_1/fc_0/BiasAdd/ReadVariableOp2N
%model_1/logits/BiasAdd/ReadVariableOp%model_1/logits/BiasAdd/ReadVariableOp2H
"model_1/fc_1/MatMul/ReadVariableOp"model_1/fc_1/MatMul/ReadVariableOp2L
$model_1/logits/MatMul/ReadVariableOp$model_1/logits/MatMul/ReadVariableOp2H
"model_1/fc_0/MatMul/ReadVariableOp"model_1/fc_0/MatMul/ReadVariableOp2J
#model_1/fc_1/BiasAdd/ReadVariableOp#model_1/fc_1/BiasAdd/ReadVariableOp:6 2
0
_user_specified_nameOvercooked_observation: : : : : : 
�
�
@__inference_logits_layer_call_and_return_conditional_losses_1240

inputs)
%matmul_readvariableop_logits_1_kernel(
$biasadd_readvariableop_logits_1_bias
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp{
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_logits_1_kernel*
dtype0*
_output_shapes

:@i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������w
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_logits_1_bias*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*.
_input_shapes
:���������@::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
�
�
A__inference_model_1_layer_call_and_return_conditional_losses_1394

inputs,
(fc_0_matmul_readvariableop_fc_0_1_kernel+
'fc_0_biasadd_readvariableop_fc_0_1_bias,
(fc_1_matmul_readvariableop_fc_1_1_kernel+
'fc_1_biasadd_readvariableop_fc_1_1_bias0
,logits_matmul_readvariableop_logits_1_kernel/
+logits_biasadd_readvariableop_logits_1_bias
identity��fc_0/BiasAdd/ReadVariableOp�fc_0/MatMul/ReadVariableOp�fc_1/BiasAdd/ReadVariableOp�fc_1/MatMul/ReadVariableOp�logits/BiasAdd/ReadVariableOp�logits/MatMul/ReadVariableOp�
fc_0/MatMul/ReadVariableOpReadVariableOp(fc_0_matmul_readvariableop_fc_0_1_kernel*
dtype0*
_output_shapes

:`@s
fc_0/MatMulMatMulinputs"fc_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@
fc_0/BiasAdd/ReadVariableOpReadVariableOp'fc_0_biasadd_readvariableop_fc_0_1_bias*
_output_shapes
:@*
dtype0�
fc_0/BiasAddBiasAddfc_0/MatMul:product:0#fc_0/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:���������@*
T0Z
	fc_0/ReluRelufc_0/BiasAdd:output:0*'
_output_shapes
:���������@*
T0�
fc_1/MatMul/ReadVariableOpReadVariableOp(fc_1_matmul_readvariableop_fc_1_1_kernel*
_output_shapes

:@@*
dtype0�
fc_1/MatMulMatMulfc_0/Relu:activations:0"fc_1/MatMul/ReadVariableOp:value:0*'
_output_shapes
:���������@*
T0
fc_1/BiasAdd/ReadVariableOpReadVariableOp'fc_1_biasadd_readvariableop_fc_1_1_bias*
dtype0*
_output_shapes
:@�
fc_1/BiasAddBiasAddfc_1/MatMul:product:0#fc_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@Z
	fc_1/ReluRelufc_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
logits/MatMul/ReadVariableOpReadVariableOp,logits_matmul_readvariableop_logits_1_kernel*
_output_shapes

:@*
dtype0�
logits/MatMulMatMulfc_1/Relu:activations:0$logits/MatMul/ReadVariableOp:value:0*'
_output_shapes
:���������*
T0�
logits/BiasAdd/ReadVariableOpReadVariableOp+logits_biasadd_readvariableop_logits_1_bias*
_output_shapes
:*
dtype0�
logits/BiasAddBiasAddlogits/MatMul:product:0%logits/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
IdentityIdentitylogits/BiasAdd:output:0^fc_0/BiasAdd/ReadVariableOp^fc_0/MatMul/ReadVariableOp^fc_1/BiasAdd/ReadVariableOp^fc_1/MatMul/ReadVariableOp^logits/BiasAdd/ReadVariableOp^logits/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*>
_input_shapes-
+:���������`::::::28
fc_0/MatMul/ReadVariableOpfc_0/MatMul/ReadVariableOp2:
fc_1/BiasAdd/ReadVariableOpfc_1/BiasAdd/ReadVariableOp2:
fc_0/BiasAdd/ReadVariableOpfc_0/BiasAdd/ReadVariableOp2>
logits/BiasAdd/ReadVariableOplogits/BiasAdd/ReadVariableOp28
fc_1/MatMul/ReadVariableOpfc_1/MatMul/ReadVariableOp2<
logits/MatMul/ReadVariableOplogits/MatMul/ReadVariableOp: : : :& "
 
_user_specified_nameinputs: : : 
�	
�
&__inference_model_1_layer_call_fn_1405

inputs)
%statefulpartitionedcall_fc_0_1_kernel'
#statefulpartitionedcall_fc_0_1_bias)
%statefulpartitionedcall_fc_1_1_kernel'
#statefulpartitionedcall_fc_1_1_bias+
'statefulpartitionedcall_logits_1_kernel)
%statefulpartitionedcall_logits_1_bias
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs%statefulpartitionedcall_fc_0_1_kernel#statefulpartitionedcall_fc_0_1_bias%statefulpartitionedcall_fc_1_1_kernel#statefulpartitionedcall_fc_1_1_bias'statefulpartitionedcall_logits_1_kernel%statefulpartitionedcall_logits_1_bias*+
_gradient_op_typePartitionedCall-1293*
Tout
2*'
_output_shapes
:���������*J
fERC
A__inference_model_1_layer_call_and_return_conditional_losses_1292*
Tin
	2**
config_proto

CPU

GPU 2J 8�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*>
_input_shapes-
+:���������`::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : :& "
 
_user_specified_nameinputs: : : 
�

�
&__inference_model_1_layer_call_fn_1330
overcooked_observation)
%statefulpartitionedcall_fc_0_1_kernel'
#statefulpartitionedcall_fc_0_1_bias)
%statefulpartitionedcall_fc_1_1_kernel'
#statefulpartitionedcall_fc_1_1_bias+
'statefulpartitionedcall_logits_1_kernel)
%statefulpartitionedcall_logits_1_bias
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallovercooked_observation%statefulpartitionedcall_fc_0_1_kernel#statefulpartitionedcall_fc_0_1_bias%statefulpartitionedcall_fc_1_1_kernel#statefulpartitionedcall_fc_1_1_bias'statefulpartitionedcall_logits_1_kernel%statefulpartitionedcall_logits_1_bias*
Tin
	2*J
fERC
A__inference_model_1_layer_call_and_return_conditional_losses_1320**
config_proto

CPU

GPU 2J 8*
Tout
2*'
_output_shapes
:���������*+
_gradient_op_typePartitionedCall-1321�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*>
_input_shapes-
+:���������`::::::22
StatefulPartitionedCallStatefulPartitionedCall:6 2
0
_user_specified_nameOvercooked_observation: : : : : : 
�
�
>__inference_fc_0_layer_call_and_return_conditional_losses_1181

inputs'
#matmul_readvariableop_fc_0_1_kernel&
"biasadd_readvariableop_fc_0_1_bias
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpy
MatMul/ReadVariableOpReadVariableOp#matmul_readvariableop_fc_0_1_kernel*
_output_shapes

:`@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@u
BiasAdd/ReadVariableOpReadVariableOp"biasadd_readvariableop_fc_0_1_bias*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:���������@*
T0P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*.
_input_shapes
:���������`::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
�
�
>__inference_fc_1_layer_call_and_return_conditional_losses_1445

inputs'
#matmul_readvariableop_fc_1_1_kernel&
"biasadd_readvariableop_fc_1_1_bias
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpy
MatMul/ReadVariableOpReadVariableOp#matmul_readvariableop_fc_1_1_kernel*
dtype0*
_output_shapes

:@@i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@u
BiasAdd/ReadVariableOpReadVariableOp"biasadd_readvariableop_fc_1_1_bias*
dtype0*
_output_shapes
:@v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*'
_output_shapes
:���������@*
T0�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:���������@*
T0"
identityIdentity:output:0*.
_input_shapes
:���������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
�	
�
"__inference_signature_wrapper_1343
overcooked_observation)
%statefulpartitionedcall_fc_0_1_kernel'
#statefulpartitionedcall_fc_0_1_bias)
%statefulpartitionedcall_fc_1_1_kernel'
#statefulpartitionedcall_fc_1_1_bias+
'statefulpartitionedcall_logits_1_kernel)
%statefulpartitionedcall_logits_1_bias
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallovercooked_observation%statefulpartitionedcall_fc_0_1_kernel#statefulpartitionedcall_fc_0_1_bias%statefulpartitionedcall_fc_1_1_kernel#statefulpartitionedcall_fc_1_1_bias'statefulpartitionedcall_logits_1_kernel%statefulpartitionedcall_logits_1_bias*'
_output_shapes
:���������*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
	2*+
_gradient_op_typePartitionedCall-1334*(
f#R!
__inference__wrapped_model_1163�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:���������*
T0"
identityIdentity:output:0*>
_input_shapes-
+:���������`::::::22
StatefulPartitionedCallStatefulPartitionedCall:6 2
0
_user_specified_nameOvercooked_observation: : : : : : 
�
�
#__inference_fc_1_layer_call_fn_1452

inputs)
%statefulpartitionedcall_fc_1_1_kernel'
#statefulpartitionedcall_fc_1_1_bias
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs%statefulpartitionedcall_fc_1_1_kernel#statefulpartitionedcall_fc_1_1_bias*
Tin
2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:���������@*+
_gradient_op_typePartitionedCall-1218*
Tout
2*G
fBR@
>__inference_fc_1_layer_call_and_return_conditional_losses_1211�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:���������@*
T0"
identityIdentity:output:0*.
_input_shapes
:���������@::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
�
�
A__inference_model_1_layer_call_and_return_conditional_losses_1276
overcooked_observation.
*fc_0_statefulpartitionedcall_fc_0_1_kernel,
(fc_0_statefulpartitionedcall_fc_0_1_bias.
*fc_1_statefulpartitionedcall_fc_1_1_kernel,
(fc_1_statefulpartitionedcall_fc_1_1_bias2
.logits_statefulpartitionedcall_logits_1_kernel0
,logits_statefulpartitionedcall_logits_1_bias
identity��fc_0/StatefulPartitionedCall�fc_1/StatefulPartitionedCall�logits/StatefulPartitionedCall�
fc_0/StatefulPartitionedCallStatefulPartitionedCallovercooked_observation*fc_0_statefulpartitionedcall_fc_0_1_kernel(fc_0_statefulpartitionedcall_fc_0_1_bias*
Tout
2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:���������@*+
_gradient_op_typePartitionedCall-1188*
Tin
2*G
fBR@
>__inference_fc_0_layer_call_and_return_conditional_losses_1181�
fc_1/StatefulPartitionedCallStatefulPartitionedCall%fc_0/StatefulPartitionedCall:output:0*fc_1_statefulpartitionedcall_fc_1_1_kernel(fc_1_statefulpartitionedcall_fc_1_1_bias*
Tin
2*'
_output_shapes
:���������@*G
fBR@
>__inference_fc_1_layer_call_and_return_conditional_losses_1211*
Tout
2*+
_gradient_op_typePartitionedCall-1218**
config_proto

CPU

GPU 2J 8�
logits/StatefulPartitionedCallStatefulPartitionedCall%fc_1/StatefulPartitionedCall:output:0.logits_statefulpartitionedcall_logits_1_kernel,logits_statefulpartitionedcall_logits_1_bias*
Tin
2*'
_output_shapes
:���������*+
_gradient_op_typePartitionedCall-1247**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_logits_layer_call_and_return_conditional_losses_1240*
Tout
2�
IdentityIdentity'logits/StatefulPartitionedCall:output:0^fc_0/StatefulPartitionedCall^fc_1/StatefulPartitionedCall^logits/StatefulPartitionedCall*'
_output_shapes
:���������*
T0"
identityIdentity:output:0*>
_input_shapes-
+:���������`::::::2<
fc_1/StatefulPartitionedCallfc_1/StatefulPartitionedCall2@
logits/StatefulPartitionedCalllogits/StatefulPartitionedCall2<
fc_0/StatefulPartitionedCallfc_0/StatefulPartitionedCall:6 2
0
_user_specified_nameOvercooked_observation: : : : : : 
�9
�
__inference__traced_save_1569
file_prefix,
(savev2_fc_0_1_kernel_read_readvariableop*
&savev2_fc_0_1_bias_read_readvariableop,
(savev2_fc_1_1_kernel_read_readvariableop*
&savev2_fc_1_1_bias_read_readvariableop.
*savev2_logits_1_kernel_read_readvariableop,
(savev2_logits_1_bias_read_readvariableop3
/savev2_training_2_adam_iter_read_readvariableop	5
1savev2_training_2_adam_beta_1_read_readvariableop5
1savev2_training_2_adam_beta_2_read_readvariableop4
0savev2_training_2_adam_decay_read_readvariableop<
8savev2_training_2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop>
:savev2_training_2_adam_fc_0_1_kernel_m_read_readvariableop<
8savev2_training_2_adam_fc_0_1_bias_m_read_readvariableop>
:savev2_training_2_adam_fc_1_1_kernel_m_read_readvariableop<
8savev2_training_2_adam_fc_1_1_bias_m_read_readvariableop@
<savev2_training_2_adam_logits_1_kernel_m_read_readvariableop>
:savev2_training_2_adam_logits_1_bias_m_read_readvariableop>
:savev2_training_2_adam_fc_0_1_kernel_v_read_readvariableop<
8savev2_training_2_adam_fc_0_1_bias_v_read_readvariableop>
:savev2_training_2_adam_fc_1_1_kernel_v_read_readvariableop<
8savev2_training_2_adam_fc_1_1_bias_v_read_readvariableop@
<savev2_training_2_adam_logits_1_kernel_v_read_readvariableop>
:savev2_training_2_adam_logits_1_bias_v_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *<
value3B1 B+_temp_6d3add39a8c14dab82c12b85cd337315/part*
dtype0s

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
value	B :*
dtype0f
ShardedFilename/shardConst"/device:CPU:0*
dtype0*
value	B : *
_output_shapes
: �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0�
SaveV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_fc_0_1_kernel_read_readvariableop&savev2_fc_0_1_bias_read_readvariableop(savev2_fc_1_1_kernel_read_readvariableop&savev2_fc_1_1_bias_read_readvariableop*savev2_logits_1_kernel_read_readvariableop(savev2_logits_1_bias_read_readvariableop/savev2_training_2_adam_iter_read_readvariableop1savev2_training_2_adam_beta_1_read_readvariableop1savev2_training_2_adam_beta_2_read_readvariableop0savev2_training_2_adam_decay_read_readvariableop8savev2_training_2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop:savev2_training_2_adam_fc_0_1_kernel_m_read_readvariableop8savev2_training_2_adam_fc_0_1_bias_m_read_readvariableop:savev2_training_2_adam_fc_1_1_kernel_m_read_readvariableop8savev2_training_2_adam_fc_1_1_bias_m_read_readvariableop<savev2_training_2_adam_logits_1_kernel_m_read_readvariableop:savev2_training_2_adam_logits_1_bias_m_read_readvariableop:savev2_training_2_adam_fc_0_1_kernel_v_read_readvariableop8savev2_training_2_adam_fc_0_1_bias_v_read_readvariableop:savev2_training_2_adam_fc_1_1_kernel_v_read_readvariableop8savev2_training_2_adam_fc_1_1_bias_v_read_readvariableop<savev2_training_2_adam_logits_1_kernel_v_read_readvariableop:savev2_training_2_adam_logits_1_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *'
dtypes
2	h
ShardedFilename_1/shardConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B :�
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
_output_shapes
:*
dtype0�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
dtypes
2*
_output_shapes
 �
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
_output_shapes
:*
T0*
N�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :`@:@:@@:@:@:: : : : : : : :`@:@:@@:@:@::`@:@:@@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1: : : : :	 :
 : : : : : : : : : : : : : : : : :+ '
%
_user_specified_namefile_prefix: : : : 
�
�
#__inference_fc_0_layer_call_fn_1434

inputs)
%statefulpartitionedcall_fc_0_1_kernel'
#statefulpartitionedcall_fc_0_1_bias
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs%statefulpartitionedcall_fc_0_1_kernel#statefulpartitionedcall_fc_0_1_bias*'
_output_shapes
:���������@*
Tout
2*
Tin
2*G
fBR@
>__inference_fc_0_layer_call_and_return_conditional_losses_1181**
config_proto

CPU

GPU 2J 8*+
_gradient_op_typePartitionedCall-1188�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*.
_input_shapes
:���������`::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : "�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*�
serving_default�
Y
Overcooked_observation?
(serving_default_Overcooked_observation:0���������`:
logits0
StatefulPartitionedCall:0���������tensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:��
�'
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api


signatures
V_default_save_signature
*W&call_and_return_all_conditional_losses
X__call__"�$
_tf_keras_model�${"class_name": "Model", "name": "model_1", "trainable": true, "expects_training_arg": true, "dtype": null, "batch_input_shape": null, "config": {"name": "model_1", "layers": [{"name": "Overcooked_observation", "class_name": "InputLayer", "config": {"batch_input_shape": [null, 96], "dtype": "float32", "sparse": false, "ragged": false, "name": "Overcooked_observation"}, "inbound_nodes": []}, {"name": "fc_0", "class_name": "Dense", "config": {"name": "fc_0", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["Overcooked_observation", 0, 0, {}]]]}, {"name": "fc_1", "class_name": "Dense", "config": {"name": "fc_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["fc_0", 0, 0, {}]]]}, {"name": "logits", "class_name": "Dense", "config": {"name": "logits", "trainable": true, "dtype": "float32", "units": 6, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["fc_1", 0, 0, {}]]]}], "input_layers": [["Overcooked_observation", 0, 0]], "output_layers": [["logits", 0, 0]]}, "input_spec": null, "activity_regularizer": null, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model_1", "layers": [{"name": "Overcooked_observation", "class_name": "InputLayer", "config": {"batch_input_shape": [null, 96], "dtype": "float32", "sparse": false, "ragged": false, "name": "Overcooked_observation"}, "inbound_nodes": []}, {"name": "fc_0", "class_name": "Dense", "config": {"name": "fc_0", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["Overcooked_observation", 0, 0, {}]]]}, {"name": "fc_1", "class_name": "Dense", "config": {"name": "fc_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["fc_0", 0, 0, {}]]]}, {"name": "logits", "class_name": "Dense", "config": {"name": "logits", "trainable": true, "dtype": "float32", "units": 6, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["fc_1", 0, 0, {}]]]}], "input_layers": [["Overcooked_observation", 0, 0]], "output_layers": [["logits", 0, 0]]}}, "training_config": {"loss": {"class_name": "SparseCategoricalCrossentropy", "config": {"reduction": "auto", "name": "sparse_categorical_crossentropy", "from_logits": true}}, "metrics": ["sparse_categorical_accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�
	variables
trainable_variables
regularization_losses
	keras_api
*Y&call_and_return_all_conditional_losses
Z__call__"�
_tf_keras_layer�{"class_name": "InputLayer", "name": "Overcooked_observation", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 96], "config": {"batch_input_shape": [null, 96], "dtype": "float32", "sparse": false, "ragged": false, "name": "Overcooked_observation"}, "input_spec": null, "activity_regularizer": null}
�

kernel
bias
_callable_losses
	variables
trainable_variables
regularization_losses
	keras_api
*[&call_and_return_all_conditional_losses
\__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "fc_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "fc_0", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 96}}}, "activity_regularizer": null}
�

kernel
bias
_callable_losses
	variables
trainable_variables
regularization_losses
	keras_api
*]&call_and_return_all_conditional_losses
^__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "fc_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "fc_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "activity_regularizer": null}
�

kernel
bias
_callable_losses
 	variables
!trainable_variables
"regularization_losses
#	keras_api
*_&call_and_return_all_conditional_losses
`__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "logits", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "logits", "trainable": true, "dtype": "float32", "units": 6, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "activity_regularizer": null}
�
$iter

%beta_1

&beta_2
	'decay
(learning_ratemJmKmLmMmNmOvPvQvRvSvTvU"
	optimizer
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
�
)layer_regularization_losses
	variables
*metrics
trainable_variables
regularization_losses

+layers
,non_trainable_variables
X__call__
V_default_save_signature
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
,
aserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
-layer_regularization_losses
	variables
.metrics
trainable_variables
regularization_losses

/layers
0non_trainable_variables
Z__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
:`@2fc_0_1/kernel
:@2fc_0_1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
1layer_regularization_losses
	variables
2metrics
trainable_variables
regularization_losses

3layers
4non_trainable_variables
\__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
:@@2fc_1_1/kernel
:@2fc_1_1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
5layer_regularization_losses
	variables
6metrics
trainable_variables
regularization_losses

7layers
8non_trainable_variables
^__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
!:@2logits_1/kernel
:2logits_1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
9layer_regularization_losses
 	variables
:metrics
!trainable_variables
"regularization_losses

;layers
<non_trainable_variables
`__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
:	 (2training_2/Adam/iter
 : (2training_2/Adam/beta_1
 : (2training_2/Adam/beta_2
: (2training_2/Adam/decay
':% (2training_2/Adam/learning_rate
 "
trackable_list_wrapper
'
=0"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
	>total
	?count
@
_fn_kwargs
A_updates
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
*b&call_and_return_all_conditional_losses
c__call__"�
_tf_keras_layer�{"class_name": "MeanMetricWrapper", "name": "sparse_categorical_accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sparse_categorical_accuracy", "dtype": "float32"}, "input_spec": null, "activity_regularizer": null}
:  (2total_1
:  (2count_1
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Flayer_regularization_losses
B	variables
Gmetrics
Ctrainable_variables
Dregularization_losses

Hlayers
Inon_trainable_variables
c__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
/:-`@2training_2/Adam/fc_0_1/kernel/m
):'@2training_2/Adam/fc_0_1/bias/m
/:-@@2training_2/Adam/fc_1_1/kernel/m
):'@2training_2/Adam/fc_1_1/bias/m
1:/@2!training_2/Adam/logits_1/kernel/m
+:)2training_2/Adam/logits_1/bias/m
/:-`@2training_2/Adam/fc_0_1/kernel/v
):'@2training_2/Adam/fc_0_1/bias/v
/:-@@2training_2/Adam/fc_1_1/kernel/v
):'@2training_2/Adam/fc_1_1/bias/v
1:/@2!training_2/Adam/logits_1/kernel/v
+:)2training_2/Adam/logits_1/bias/v
�2�
__inference__wrapped_model_1163�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *5�2
0�-
Overcooked_observation���������`
�2�
A__inference_model_1_layer_call_and_return_conditional_losses_1394
A__inference_model_1_layer_call_and_return_conditional_losses_1276
A__inference_model_1_layer_call_and_return_conditional_losses_1370
A__inference_model_1_layer_call_and_return_conditional_losses_1260�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
&__inference_model_1_layer_call_fn_1416
&__inference_model_1_layer_call_fn_1405
&__inference_model_1_layer_call_fn_1330
&__inference_model_1_layer_call_fn_1302�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2�
>__inference_fc_0_layer_call_and_return_conditional_losses_1427�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
#__inference_fc_0_layer_call_fn_1434�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
>__inference_fc_1_layer_call_and_return_conditional_losses_1445�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
#__inference_fc_1_layer_call_fn_1452�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
@__inference_logits_layer_call_and_return_conditional_losses_1462�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
%__inference_logits_layer_call_fn_1469�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
@B>
"__inference_signature_wrapper_1343Overcooked_observation
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 �
A__inference_model_1_layer_call_and_return_conditional_losses_1276xG�D
=�:
0�-
Overcooked_observation���������`
p 

 
� "%�"
�
0���������
� �
A__inference_model_1_layer_call_and_return_conditional_losses_1394h7�4
-�*
 �
inputs���������`
p 

 
� "%�"
�
0���������
� �
A__inference_model_1_layer_call_and_return_conditional_losses_1260xG�D
=�:
0�-
Overcooked_observation���������`
p

 
� "%�"
�
0���������
� v
#__inference_fc_1_layer_call_fn_1452O/�,
%�"
 �
inputs���������@
� "����������@�
>__inference_fc_1_layer_call_and_return_conditional_losses_1445\/�,
%�"
 �
inputs���������@
� "%�"
�
0���������@
� �
&__inference_model_1_layer_call_fn_1416[7�4
-�*
 �
inputs���������`
p 

 
� "�����������
__inference__wrapped_model_1163z?�<
5�2
0�-
Overcooked_observation���������`
� "/�,
*
logits �
logits����������
&__inference_model_1_layer_call_fn_1302kG�D
=�:
0�-
Overcooked_observation���������`
p

 
� "����������x
%__inference_logits_layer_call_fn_1469O/�,
%�"
 �
inputs���������@
� "�����������
A__inference_model_1_layer_call_and_return_conditional_losses_1370h7�4
-�*
 �
inputs���������`
p

 
� "%�"
�
0���������
� �
&__inference_model_1_layer_call_fn_1330kG�D
=�:
0�-
Overcooked_observation���������`
p 

 
� "�����������
@__inference_logits_layer_call_and_return_conditional_losses_1462\/�,
%�"
 �
inputs���������@
� "%�"
�
0���������
� �
&__inference_model_1_layer_call_fn_1405[7�4
-�*
 �
inputs���������`
p

 
� "�����������
"__inference_signature_wrapper_1343�Y�V
� 
O�L
J
Overcooked_observation0�-
Overcooked_observation���������`"/�,
*
logits �
logits���������v
#__inference_fc_0_layer_call_fn_1434O/�,
%�"
 �
inputs���������`
� "����������@�
>__inference_fc_0_layer_call_and_return_conditional_losses_1427\/�,
%�"
 �
inputs���������`
� "%�"
�
0���������@
� 