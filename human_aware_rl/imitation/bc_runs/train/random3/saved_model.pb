╓╢
Ъ¤
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
dtypetypeИ
╛
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
executor_typestring И
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeИ"serve*1.15.02unknown8НШ
r
fc_0/kernelVarHandleOp*
dtype0*
shared_namefc_0/kernel*
shape
:`@*
_output_shapes
: 
k
fc_0/kernel/Read/ReadVariableOpReadVariableOpfc_0/kernel*
_output_shapes

:`@*
dtype0
j
	fc_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name	fc_0/bias
c
fc_0/bias/Read/ReadVariableOpReadVariableOp	fc_0/bias*
dtype0*
_output_shapes
:@
r
fc_1/kernelVarHandleOp*
shape
:@@*
_output_shapes
: *
dtype0*
shared_namefc_1/kernel
k
fc_1/kernel/Read/ReadVariableOpReadVariableOpfc_1/kernel*
_output_shapes

:@@*
dtype0
j
	fc_1/biasVarHandleOp*
dtype0*
_output_shapes
: *
shared_name	fc_1/bias*
shape:@
c
fc_1/bias/Read/ReadVariableOpReadVariableOp	fc_1/bias*
_output_shapes
:@*
dtype0
v
logits/kernelVarHandleOp*
shape
:@*
_output_shapes
: *
shared_namelogits/kernel*
dtype0
o
!logits/kernel/Read/ReadVariableOpReadVariableOplogits/kernel*
_output_shapes

:@*
dtype0
n
logits/biasVarHandleOp*
dtype0*
shape:*
_output_shapes
: *
shared_namelogits/bias
g
logits/bias/Read/ReadVariableOpReadVariableOplogits/bias*
_output_shapes
:*
dtype0
x
training/Adam/iterVarHandleOp*#
shared_nametraining/Adam/iter*
dtype0	*
_output_shapes
: *
shape: 
q
&training/Adam/iter/Read/ReadVariableOpReadVariableOptraining/Adam/iter*
dtype0	*
_output_shapes
: 
|
training/Adam/beta_1VarHandleOp*
_output_shapes
: *%
shared_nametraining/Adam/beta_1*
shape: *
dtype0
u
(training/Adam/beta_1/Read/ReadVariableOpReadVariableOptraining/Adam/beta_1*
_output_shapes
: *
dtype0
|
training/Adam/beta_2VarHandleOp*%
shared_nametraining/Adam/beta_2*
dtype0*
_output_shapes
: *
shape: 
u
(training/Adam/beta_2/Read/ReadVariableOpReadVariableOptraining/Adam/beta_2*
dtype0*
_output_shapes
: 
z
training/Adam/decayVarHandleOp*$
shared_nametraining/Adam/decay*
shape: *
dtype0*
_output_shapes
: 
s
'training/Adam/decay/Read/ReadVariableOpReadVariableOptraining/Adam/decay*
dtype0*
_output_shapes
: 
К
training/Adam/learning_rateVarHandleOp*,
shared_nametraining/Adam/learning_rate*
shape: *
dtype0*
_output_shapes
: 
Г
/training/Adam/learning_rate/Read/ReadVariableOpReadVariableOptraining/Adam/learning_rate*
dtype0*
_output_shapes
: 
^
totalVarHandleOp*
dtype0*
shared_nametotal*
shape: *
_output_shapes
: 
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
shape: *
shared_namecount*
dtype0*
_output_shapes
: 
W
count/Read/ReadVariableOpReadVariableOpcount*
dtype0*
_output_shapes
: 
Т
training/Adam/fc_0/kernel/mVarHandleOp*
dtype0*
shape
:`@*
_output_shapes
: *,
shared_nametraining/Adam/fc_0/kernel/m
Л
/training/Adam/fc_0/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/fc_0/kernel/m*
_output_shapes

:`@*
dtype0
К
training/Adam/fc_0/bias/mVarHandleOp**
shared_nametraining/Adam/fc_0/bias/m*
shape:@*
dtype0*
_output_shapes
: 
Г
-training/Adam/fc_0/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/fc_0/bias/m*
dtype0*
_output_shapes
:@
Т
training/Adam/fc_1/kernel/mVarHandleOp*
_output_shapes
: *,
shared_nametraining/Adam/fc_1/kernel/m*
shape
:@@*
dtype0
Л
/training/Adam/fc_1/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/fc_1/kernel/m*
_output_shapes

:@@*
dtype0
К
training/Adam/fc_1/bias/mVarHandleOp*
_output_shapes
: *
shape:@**
shared_nametraining/Adam/fc_1/bias/m*
dtype0
Г
-training/Adam/fc_1/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/fc_1/bias/m*
dtype0*
_output_shapes
:@
Ц
training/Adam/logits/kernel/mVarHandleOp*
dtype0*
shape
:@*
_output_shapes
: *.
shared_nametraining/Adam/logits/kernel/m
П
1training/Adam/logits/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/logits/kernel/m*
dtype0*
_output_shapes

:@
О
training/Adam/logits/bias/mVarHandleOp*
dtype0*
_output_shapes
: *
shape:*,
shared_nametraining/Adam/logits/bias/m
З
/training/Adam/logits/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/logits/bias/m*
_output_shapes
:*
dtype0
Т
training/Adam/fc_0/kernel/vVarHandleOp*
dtype0*,
shared_nametraining/Adam/fc_0/kernel/v*
shape
:`@*
_output_shapes
: 
Л
/training/Adam/fc_0/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/fc_0/kernel/v*
dtype0*
_output_shapes

:`@
К
training/Adam/fc_0/bias/vVarHandleOp**
shared_nametraining/Adam/fc_0/bias/v*
_output_shapes
: *
shape:@*
dtype0
Г
-training/Adam/fc_0/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/fc_0/bias/v*
_output_shapes
:@*
dtype0
Т
training/Adam/fc_1/kernel/vVarHandleOp*
shape
:@@*,
shared_nametraining/Adam/fc_1/kernel/v*
dtype0*
_output_shapes
: 
Л
/training/Adam/fc_1/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/fc_1/kernel/v*
_output_shapes

:@@*
dtype0
К
training/Adam/fc_1/bias/vVarHandleOp**
shared_nametraining/Adam/fc_1/bias/v*
_output_shapes
: *
shape:@*
dtype0
Г
-training/Adam/fc_1/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/fc_1/bias/v*
_output_shapes
:@*
dtype0
Ц
training/Adam/logits/kernel/vVarHandleOp*
dtype0*
_output_shapes
: *
shape
:@*.
shared_nametraining/Adam/logits/kernel/v
П
1training/Adam/logits/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/logits/kernel/v*
_output_shapes

:@*
dtype0
О
training/Adam/logits/bias/vVarHandleOp*
dtype0*
shape:*,
shared_nametraining/Adam/logits/bias/v*
_output_shapes
: 
З
/training/Adam/logits/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/logits/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
ї'
ConstConst"/device:CPU:0*
dtype0*░'
valueж'Bг' BЬ'
є
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
м
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
Ъ
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
Ъ
-layer_regularization_losses
	variables
.metrics
trainable_variables
regularization_losses

/layers
0non_trainable_variables
WU
VARIABLE_VALUEfc_0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	fc_0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
 
Ъ
1layer_regularization_losses
	variables
2metrics
trainable_variables
regularization_losses

3layers
4non_trainable_variables
WU
VARIABLE_VALUEfc_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	fc_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
 
Ъ
5layer_regularization_losses
	variables
6metrics
trainable_variables
regularization_losses

7layers
8non_trainable_variables
YW
VARIABLE_VALUElogits/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElogits/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
 
Ъ
9layer_regularization_losses
 	variables
:metrics
!trainable_variables
"regularization_losses

;layers
<non_trainable_variables
QO
VARIABLE_VALUEtraining/Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEtraining/Adam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEtraining/Adam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEtraining/Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEtraining/Adam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
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
Ж
	>total
	?count
@
_fn_kwargs
A_updates
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 

>0
?1
 
 
Ъ
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
ДБ
VARIABLE_VALUEtraining/Adam/fc_0/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEtraining/Adam/fc_0/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ДБ
VARIABLE_VALUEtraining/Adam/fc_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEtraining/Adam/fc_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЖГ
VARIABLE_VALUEtraining/Adam/logits/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUEtraining/Adam/logits/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ДБ
VARIABLE_VALUEtraining/Adam/fc_0/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEtraining/Adam/fc_0/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ДБ
VARIABLE_VALUEtraining/Adam/fc_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEtraining/Adam/fc_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЖГ
VARIABLE_VALUEtraining/Adam/logits/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUEtraining/Adam/logits/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
_output_shapes
: 
Й
&serving_default_Overcooked_observationPlaceholder*
shape:         `*
dtype0*'
_output_shapes
:         `
Ў
StatefulPartitionedCallStatefulPartitionedCall&serving_default_Overcooked_observationfc_0/kernel	fc_0/biasfc_1/kernel	fc_1/biaslogits/kernellogits/bias**
_gradient_op_typePartitionedCall-687*
Tout
2**
f%R#
!__inference_signature_wrapper_507**
config_proto

CPU

GPU 2J 8*
Tin
	2*'
_output_shapes
:         
O
saver_filenamePlaceholder*
_output_shapes
: *
shape: *
dtype0
▒

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamefc_0/kernel/Read/ReadVariableOpfc_0/bias/Read/ReadVariableOpfc_1/kernel/Read/ReadVariableOpfc_1/bias/Read/ReadVariableOp!logits/kernel/Read/ReadVariableOplogits/bias/Read/ReadVariableOp&training/Adam/iter/Read/ReadVariableOp(training/Adam/beta_1/Read/ReadVariableOp(training/Adam/beta_2/Read/ReadVariableOp'training/Adam/decay/Read/ReadVariableOp/training/Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp/training/Adam/fc_0/kernel/m/Read/ReadVariableOp-training/Adam/fc_0/bias/m/Read/ReadVariableOp/training/Adam/fc_1/kernel/m/Read/ReadVariableOp-training/Adam/fc_1/bias/m/Read/ReadVariableOp1training/Adam/logits/kernel/m/Read/ReadVariableOp/training/Adam/logits/bias/m/Read/ReadVariableOp/training/Adam/fc_0/kernel/v/Read/ReadVariableOp-training/Adam/fc_0/bias/v/Read/ReadVariableOp/training/Adam/fc_1/kernel/v/Read/ReadVariableOp-training/Adam/fc_1/bias/v/Read/ReadVariableOp1training/Adam/logits/kernel/v/Read/ReadVariableOp/training/Adam/logits/bias/v/Read/ReadVariableOpConst**
config_proto

CPU

GPU 2J 8*&
Tin
2	**
_gradient_op_typePartitionedCall-734*
Tout
2*%
f R
__inference__traced_save_733*
_output_shapes
: 
╕
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamefc_0/kernel	fc_0/biasfc_1/kernel	fc_1/biaslogits/kernellogits/biastraining/Adam/itertraining/Adam/beta_1training/Adam/beta_2training/Adam/decaytraining/Adam/learning_ratetotalcounttraining/Adam/fc_0/kernel/mtraining/Adam/fc_0/bias/mtraining/Adam/fc_1/kernel/mtraining/Adam/fc_1/bias/mtraining/Adam/logits/kernel/mtraining/Adam/logits/bias/mtraining/Adam/fc_0/kernel/vtraining/Adam/fc_0/bias/vtraining/Adam/fc_1/kernel/vtraining/Adam/fc_1/bias/vtraining/Adam/logits/kernel/vtraining/Adam/logits/bias/v*%
Tin
2*
_output_shapes
: **
config_proto

CPU

GPU 2J 8**
_gradient_op_typePartitionedCall-822*
Tout
2*(
f#R!
__inference__traced_restore_821ри
є
┌
=__inference_fc_0_layer_call_and_return_conditional_losses_345

inputs%
!matmul_readvariableop_fc_0_kernel$
 biasadd_readvariableop_fc_0_bias
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpw
MatMul/ReadVariableOpReadVariableOp!matmul_readvariableop_fc_0_kernel*
_output_shapes

:`@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @s
BiasAdd/ReadVariableOpReadVariableOp biasadd_readvariableop_fc_0_bias*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:         @*
T0P
ReluReluBiasAdd:output:0*'
_output_shapes
:         @*
T0Л
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*.
_input_shapes
:         `::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
у
▒
$__inference_logits_layer_call_fn_633

inputs)
%statefulpartitionedcall_logits_kernel'
#statefulpartitionedcall_logits_bias
identityИвStatefulPartitionedCallЁ
StatefulPartitionedCallStatefulPartitionedCallinputs%statefulpartitionedcall_logits_kernel#statefulpartitionedcall_logits_bias*
Tout
2*'
_output_shapes
:         *H
fCRA
?__inference_logits_layer_call_and_return_conditional_losses_404**
_gradient_op_typePartitionedCall-411**
config_proto

CPU

GPU 2J 8*
Tin
2В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*.
_input_shapes
:         @::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
╦	
▐
!__inference_signature_wrapper_507
overcooked_observation'
#statefulpartitionedcall_fc_0_kernel%
!statefulpartitionedcall_fc_0_bias'
#statefulpartitionedcall_fc_1_kernel%
!statefulpartitionedcall_fc_1_bias)
%statefulpartitionedcall_logits_kernel'
#statefulpartitionedcall_logits_bias
identityИвStatefulPartitionedCallє
StatefulPartitionedCallStatefulPartitionedCallovercooked_observation#statefulpartitionedcall_fc_0_kernel!statefulpartitionedcall_fc_0_bias#statefulpartitionedcall_fc_1_kernel!statefulpartitionedcall_fc_1_bias%statefulpartitionedcall_logits_kernel#statefulpartitionedcall_logits_bias*
Tin
	2**
config_proto

CPU

GPU 2J 8**
_gradient_op_typePartitionedCall-498*'
_output_shapes
:         *
Tout
2*'
f"R 
__inference__wrapped_model_327В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*>
_input_shapes-
+:         `::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : :6 2
0
_user_specified_nameOvercooked_observation: : : 
┴
Я
>__inference_model_layer_call_and_return_conditional_losses_534

inputs*
&fc_0_matmul_readvariableop_fc_0_kernel)
%fc_0_biasadd_readvariableop_fc_0_bias*
&fc_1_matmul_readvariableop_fc_1_kernel)
%fc_1_biasadd_readvariableop_fc_1_bias.
*logits_matmul_readvariableop_logits_kernel-
)logits_biasadd_readvariableop_logits_bias
identityИвfc_0/BiasAdd/ReadVariableOpвfc_0/MatMul/ReadVariableOpвfc_1/BiasAdd/ReadVariableOpвfc_1/MatMul/ReadVariableOpвlogits/BiasAdd/ReadVariableOpвlogits/MatMul/ReadVariableOpБ
fc_0/MatMul/ReadVariableOpReadVariableOp&fc_0_matmul_readvariableop_fc_0_kernel*
_output_shapes

:`@*
dtype0s
fc_0/MatMulMatMulinputs"fc_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @}
fc_0/BiasAdd/ReadVariableOpReadVariableOp%fc_0_biasadd_readvariableop_fc_0_bias*
dtype0*
_output_shapes
:@Е
fc_0/BiasAddBiasAddfc_0/MatMul:product:0#fc_0/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:         @*
T0Z
	fc_0/ReluRelufc_0/BiasAdd:output:0*
T0*'
_output_shapes
:         @Б
fc_1/MatMul/ReadVariableOpReadVariableOp&fc_1_matmul_readvariableop_fc_1_kernel*
dtype0*
_output_shapes

:@@Д
fc_1/MatMulMatMulfc_0/Relu:activations:0"fc_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @}
fc_1/BiasAdd/ReadVariableOpReadVariableOp%fc_1_biasadd_readvariableop_fc_1_bias*
_output_shapes
:@*
dtype0Е
fc_1/BiasAddBiasAddfc_1/MatMul:product:0#fc_1/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:         @*
T0Z
	fc_1/ReluRelufc_1/BiasAdd:output:0*
T0*'
_output_shapes
:         @З
logits/MatMul/ReadVariableOpReadVariableOp*logits_matmul_readvariableop_logits_kernel*
_output_shapes

:@*
dtype0И
logits/MatMulMatMulfc_1/Relu:activations:0$logits/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Г
logits/BiasAdd/ReadVariableOpReadVariableOp)logits_biasadd_readvariableop_logits_bias*
dtype0*
_output_shapes
:Л
logits/BiasAddBiasAddlogits/MatMul:product:0%logits/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ф
IdentityIdentitylogits/BiasAdd:output:0^fc_0/BiasAdd/ReadVariableOp^fc_0/MatMul/ReadVariableOp^fc_1/BiasAdd/ReadVariableOp^fc_1/MatMul/ReadVariableOp^logits/BiasAdd/ReadVariableOp^logits/MatMul/ReadVariableOp*'
_output_shapes
:         *
T0"
identityIdentity:output:0*>
_input_shapes-
+:         `::::::28
fc_0/MatMul/ReadVariableOpfc_0/MatMul/ReadVariableOp2:
fc_1/BiasAdd/ReadVariableOpfc_1/BiasAdd/ReadVariableOp2:
fc_0/BiasAdd/ReadVariableOpfc_0/BiasAdd/ReadVariableOp2>
logits/BiasAdd/ReadVariableOplogits/BiasAdd/ReadVariableOp28
fc_1/MatMul/ReadVariableOpfc_1/MatMul/ReadVariableOp2<
logits/MatMul/ReadVariableOplogits/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : : : 
й
р
?__inference_logits_layer_call_and_return_conditional_losses_626

inputs'
#matmul_readvariableop_logits_kernel&
"biasadd_readvariableop_logits_bias
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpy
MatMul/ReadVariableOpReadVariableOp#matmul_readvariableop_logits_kernel*
dtype0*
_output_shapes

:@i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         u
BiasAdd/ReadVariableOpReadVariableOp"biasadd_readvariableop_logits_bias*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:         *
T0Й
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:         *
T0"
identityIdentity:output:0*.
_input_shapes
:         @::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
═
╥
>__inference_model_layer_call_and_return_conditional_losses_484

inputs,
(fc_0_statefulpartitionedcall_fc_0_kernel*
&fc_0_statefulpartitionedcall_fc_0_bias,
(fc_1_statefulpartitionedcall_fc_1_kernel*
&fc_1_statefulpartitionedcall_fc_1_bias0
,logits_statefulpartitionedcall_logits_kernel.
*logits_statefulpartitionedcall_logits_bias
identityИвfc_0/StatefulPartitionedCallвfc_1/StatefulPartitionedCallвlogits/StatefulPartitionedCall∙
fc_0/StatefulPartitionedCallStatefulPartitionedCallinputs(fc_0_statefulpartitionedcall_fc_0_kernel&fc_0_statefulpartitionedcall_fc_0_bias**
config_proto

CPU

GPU 2J 8*F
fAR?
=__inference_fc_0_layer_call_and_return_conditional_losses_345*
Tin
2*
Tout
2**
_gradient_op_typePartitionedCall-352*'
_output_shapes
:         @Ш
fc_1/StatefulPartitionedCallStatefulPartitionedCall%fc_0/StatefulPartitionedCall:output:0(fc_1_statefulpartitionedcall_fc_1_kernel&fc_1_statefulpartitionedcall_fc_1_bias**
config_proto

CPU

GPU 2J 8*F
fAR?
=__inference_fc_1_layer_call_and_return_conditional_losses_375*
Tout
2*
Tin
2*'
_output_shapes
:         @**
_gradient_op_typePartitionedCall-382д
logits/StatefulPartitionedCallStatefulPartitionedCall%fc_1/StatefulPartitionedCall:output:0,logits_statefulpartitionedcall_logits_kernel*logits_statefulpartitionedcall_logits_bias**
config_proto

CPU

GPU 2J 8*
Tout
2*H
fCRA
?__inference_logits_layer_call_and_return_conditional_losses_404*'
_output_shapes
:         **
_gradient_op_typePartitionedCall-411*
Tin
2╬
IdentityIdentity'logits/StatefulPartitionedCall:output:0^fc_0/StatefulPartitionedCall^fc_1/StatefulPartitionedCall^logits/StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*>
_input_shapes-
+:         `::::::2<
fc_0/StatefulPartitionedCallfc_0/StatefulPartitionedCall2<
fc_1/StatefulPartitionedCallfc_1/StatefulPartitionedCall2@
logits/StatefulPartitionedCalllogits/StatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : : : : 
╜	
╨
#__inference_model_layer_call_fn_580

inputs'
#statefulpartitionedcall_fc_0_kernel%
!statefulpartitionedcall_fc_0_bias'
#statefulpartitionedcall_fc_1_kernel%
!statefulpartitionedcall_fc_1_bias)
%statefulpartitionedcall_logits_kernel'
#statefulpartitionedcall_logits_bias
identityИвStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputs#statefulpartitionedcall_fc_0_kernel!statefulpartitionedcall_fc_0_bias#statefulpartitionedcall_fc_1_kernel!statefulpartitionedcall_fc_1_bias%statefulpartitionedcall_logits_kernel#statefulpartitionedcall_logits_bias**
config_proto

CPU

GPU 2J 8*
Tout
2**
_gradient_op_typePartitionedCall-485*'
_output_shapes
:         *
Tin
	2*G
fBR@
>__inference_model_layer_call_and_return_conditional_losses_484В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*>
_input_shapes-
+:         `::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :& "
 
_user_specified_nameinputs: : 
э	
р
#__inference_model_layer_call_fn_466
overcooked_observation'
#statefulpartitionedcall_fc_0_kernel%
!statefulpartitionedcall_fc_0_bias'
#statefulpartitionedcall_fc_1_kernel%
!statefulpartitionedcall_fc_1_bias)
%statefulpartitionedcall_logits_kernel'
#statefulpartitionedcall_logits_bias
identityИвStatefulPartitionedCallУ
StatefulPartitionedCallStatefulPartitionedCallovercooked_observation#statefulpartitionedcall_fc_0_kernel!statefulpartitionedcall_fc_0_bias#statefulpartitionedcall_fc_1_kernel!statefulpartitionedcall_fc_1_bias%statefulpartitionedcall_logits_kernel#statefulpartitionedcall_logits_bias*'
_output_shapes
:         **
config_proto

CPU

GPU 2J 8**
_gradient_op_typePartitionedCall-457*
Tout
2*
Tin
	2*G
fBR@
>__inference_model_layer_call_and_return_conditional_losses_456В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:         *
T0"
identityIdentity:output:0*>
_input_shapes-
+:         `::::::22
StatefulPartitionedCallStatefulPartitionedCall:6 2
0
_user_specified_nameOvercooked_observation: : : : : : 
╫
л
"__inference_fc_0_layer_call_fn_598

inputs'
#statefulpartitionedcall_fc_0_kernel%
!statefulpartitionedcall_fc_0_bias
identityИвStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputs#statefulpartitionedcall_fc_0_kernel!statefulpartitionedcall_fc_0_bias*'
_output_shapes
:         @*F
fAR?
=__inference_fc_0_layer_call_and_return_conditional_losses_345**
config_proto

CPU

GPU 2J 8**
_gradient_op_typePartitionedCall-352*
Tout
2*
Tin
2В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:         @*
T0"
identityIdentity:output:0*.
_input_shapes
:         `::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
╜	
╨
#__inference_model_layer_call_fn_569

inputs'
#statefulpartitionedcall_fc_0_kernel%
!statefulpartitionedcall_fc_0_bias'
#statefulpartitionedcall_fc_1_kernel%
!statefulpartitionedcall_fc_1_bias)
%statefulpartitionedcall_logits_kernel'
#statefulpartitionedcall_logits_bias
identityИвStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputs#statefulpartitionedcall_fc_0_kernel!statefulpartitionedcall_fc_0_bias#statefulpartitionedcall_fc_1_kernel!statefulpartitionedcall_fc_1_bias%statefulpartitionedcall_logits_kernel#statefulpartitionedcall_logits_bias*
Tout
2**
config_proto

CPU

GPU 2J 8*G
fBR@
>__inference_model_layer_call_and_return_conditional_losses_456*'
_output_shapes
:         *
Tin
	2**
_gradient_op_typePartitionedCall-457В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:         *
T0"
identityIdentity:output:0*>
_input_shapes-
+:         `::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : 
¤
т
>__inference_model_layer_call_and_return_conditional_losses_440
overcooked_observation,
(fc_0_statefulpartitionedcall_fc_0_kernel*
&fc_0_statefulpartitionedcall_fc_0_bias,
(fc_1_statefulpartitionedcall_fc_1_kernel*
&fc_1_statefulpartitionedcall_fc_1_bias0
,logits_statefulpartitionedcall_logits_kernel.
*logits_statefulpartitionedcall_logits_bias
identityИвfc_0/StatefulPartitionedCallвfc_1/StatefulPartitionedCallвlogits/StatefulPartitionedCallЙ
fc_0/StatefulPartitionedCallStatefulPartitionedCallovercooked_observation(fc_0_statefulpartitionedcall_fc_0_kernel&fc_0_statefulpartitionedcall_fc_0_bias*'
_output_shapes
:         @*
Tin
2**
_gradient_op_typePartitionedCall-352**
config_proto

CPU

GPU 2J 8*F
fAR?
=__inference_fc_0_layer_call_and_return_conditional_losses_345*
Tout
2Ш
fc_1/StatefulPartitionedCallStatefulPartitionedCall%fc_0/StatefulPartitionedCall:output:0(fc_1_statefulpartitionedcall_fc_1_kernel&fc_1_statefulpartitionedcall_fc_1_bias**
_gradient_op_typePartitionedCall-382*
Tin
2*
Tout
2**
config_proto

CPU

GPU 2J 8*F
fAR?
=__inference_fc_1_layer_call_and_return_conditional_losses_375*'
_output_shapes
:         @д
logits/StatefulPartitionedCallStatefulPartitionedCall%fc_1/StatefulPartitionedCall:output:0,logits_statefulpartitionedcall_logits_kernel*logits_statefulpartitionedcall_logits_bias*
Tout
2*'
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*
Tin
2**
_gradient_op_typePartitionedCall-411*H
fCRA
?__inference_logits_layer_call_and_return_conditional_losses_404╬
IdentityIdentity'logits/StatefulPartitionedCall:output:0^fc_0/StatefulPartitionedCall^fc_1/StatefulPartitionedCall^logits/StatefulPartitionedCall*'
_output_shapes
:         *
T0"
identityIdentity:output:0*>
_input_shapes-
+:         `::::::2<
fc_0/StatefulPartitionedCallfc_0/StatefulPartitionedCall2<
fc_1/StatefulPartitionedCallfc_1/StatefulPartitionedCall2@
logits/StatefulPartitionedCalllogits/StatefulPartitionedCall:6 2
0
_user_specified_nameOvercooked_observation: : : : : : 
╫
л
"__inference_fc_1_layer_call_fn_616

inputs'
#statefulpartitionedcall_fc_1_kernel%
!statefulpartitionedcall_fc_1_bias
identityИвStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputs#statefulpartitionedcall_fc_1_kernel!statefulpartitionedcall_fc_1_bias*
Tout
2*
Tin
2*'
_output_shapes
:         @*F
fAR?
=__inference_fc_1_layer_call_and_return_conditional_losses_375**
config_proto

CPU

GPU 2J 8**
_gradient_op_typePartitionedCall-382В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:         @*
T0"
identityIdentity:output:0*.
_input_shapes
:         @::22
StatefulPartitionedCallStatefulPartitionedCall: : :& "
 
_user_specified_nameinputs
є
┌
=__inference_fc_1_layer_call_and_return_conditional_losses_375

inputs%
!matmul_readvariableop_fc_1_kernel$
 biasadd_readvariableop_fc_1_bias
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpw
MatMul/ReadVariableOpReadVariableOp!matmul_readvariableop_fc_1_kernel*
dtype0*
_output_shapes

:@@i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @s
BiasAdd/ReadVariableOpReadVariableOp biasadd_readvariableop_fc_1_bias*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         @Л
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:         @*
T0"
identityIdentity:output:0*.
_input_shapes
:         @::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
Ж8
к
__inference__traced_save_733
file_prefix*
&savev2_fc_0_kernel_read_readvariableop(
$savev2_fc_0_bias_read_readvariableop*
&savev2_fc_1_kernel_read_readvariableop(
$savev2_fc_1_bias_read_readvariableop,
(savev2_logits_kernel_read_readvariableop*
&savev2_logits_bias_read_readvariableop1
-savev2_training_adam_iter_read_readvariableop	3
/savev2_training_adam_beta_1_read_readvariableop3
/savev2_training_adam_beta_2_read_readvariableop2
.savev2_training_adam_decay_read_readvariableop:
6savev2_training_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop:
6savev2_training_adam_fc_0_kernel_m_read_readvariableop8
4savev2_training_adam_fc_0_bias_m_read_readvariableop:
6savev2_training_adam_fc_1_kernel_m_read_readvariableop8
4savev2_training_adam_fc_1_bias_m_read_readvariableop<
8savev2_training_adam_logits_kernel_m_read_readvariableop:
6savev2_training_adam_logits_bias_m_read_readvariableop:
6savev2_training_adam_fc_0_kernel_v_read_readvariableop8
4savev2_training_adam_fc_0_bias_v_read_readvariableop:
6savev2_training_adam_fc_1_kernel_v_read_readvariableop8
4savev2_training_adam_fc_1_bias_v_read_readvariableop<
8savev2_training_adam_logits_kernel_v_read_readvariableop:
6savev2_training_adam_logits_bias_v_read_readvariableop
savev2_1_const

identity_1ИвMergeV2CheckpointsвSaveV2вSaveV2_1О
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_5eef27a6f9fb4592b6040557cf994a56/part*
_output_shapes
: *
dtype0s

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
_output_shapes
: *
NL

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ы
SaveV2/tensor_namesConst"/device:CPU:0*
dtype0*Ф
valueКBЗB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
_output_shapes
:Я
SaveV2/shape_and_slicesConst"/device:CPU:0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:*
dtype0Б
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0&savev2_fc_0_kernel_read_readvariableop$savev2_fc_0_bias_read_readvariableop&savev2_fc_1_kernel_read_readvariableop$savev2_fc_1_bias_read_readvariableop(savev2_logits_kernel_read_readvariableop&savev2_logits_bias_read_readvariableop-savev2_training_adam_iter_read_readvariableop/savev2_training_adam_beta_1_read_readvariableop/savev2_training_adam_beta_2_read_readvariableop.savev2_training_adam_decay_read_readvariableop6savev2_training_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop6savev2_training_adam_fc_0_kernel_m_read_readvariableop4savev2_training_adam_fc_0_bias_m_read_readvariableop6savev2_training_adam_fc_1_kernel_m_read_readvariableop4savev2_training_adam_fc_1_bias_m_read_readvariableop8savev2_training_adam_logits_kernel_m_read_readvariableop6savev2_training_adam_logits_bias_m_read_readvariableop6savev2_training_adam_fc_0_kernel_v_read_readvariableop4savev2_training_adam_fc_0_bias_v_read_readvariableop6savev2_training_adam_fc_1_kernel_v_read_readvariableop4savev2_training_adam_fc_1_bias_v_read_readvariableop8savev2_training_adam_logits_kernel_v_read_readvariableop6savev2_training_adam_logits_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *'
dtypes
2	h
ShardedFilename_1/shardConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B :Ч
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Й
SaveV2_1/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPHq
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
_output_shapes
:*
dtype0├
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
2╣
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:Ц
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

identity_1Identity_1:output:0*╖
_input_shapesе
в: :`@:@:@@:@:@:: : : : : : : :`@:@:@@:@:@::`@:@:@@:@:@:: 2
SaveV2_1SaveV2_12
SaveV2SaveV22(
MergeV2CheckpointsMergeV2Checkpoints:+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : 
▌c
ъ
__inference__traced_restore_821
file_prefix 
assignvariableop_fc_0_kernel 
assignvariableop_1_fc_0_bias"
assignvariableop_2_fc_1_kernel 
assignvariableop_3_fc_1_bias$
 assignvariableop_4_logits_kernel"
assignvariableop_5_logits_bias)
%assignvariableop_6_training_adam_iter+
'assignvariableop_7_training_adam_beta_1+
'assignvariableop_8_training_adam_beta_2*
&assignvariableop_9_training_adam_decay3
/assignvariableop_10_training_adam_learning_rate
assignvariableop_11_total
assignvariableop_12_count3
/assignvariableop_13_training_adam_fc_0_kernel_m1
-assignvariableop_14_training_adam_fc_0_bias_m3
/assignvariableop_15_training_adam_fc_1_kernel_m1
-assignvariableop_16_training_adam_fc_1_bias_m5
1assignvariableop_17_training_adam_logits_kernel_m3
/assignvariableop_18_training_adam_logits_bias_m3
/assignvariableop_19_training_adam_fc_0_kernel_v1
-assignvariableop_20_training_adam_fc_0_bias_v3
/assignvariableop_21_training_adam_fc_1_kernel_v1
-assignvariableop_22_training_adam_fc_1_bias_v5
1assignvariableop_23_training_adam_logits_kernel_v3
/assignvariableop_24_training_adam_logits_bias_v
identity_26ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9в	RestoreV2вRestoreV2_1ю
RestoreV2/tensor_namesConst"/device:CPU:0*Ф
valueКBЗB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:в
RestoreV2/shape_and_slicesConst"/device:CPU:0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:*
dtype0Ы
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*'
dtypes
2	*x
_output_shapesf
d:::::::::::::::::::::::::L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:x
AssignVariableOpAssignVariableOpassignvariableop_fc_0_kernelIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
_output_shapes
:*
T0|
AssignVariableOp_1AssignVariableOpassignvariableop_1_fc_0_biasIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:~
AssignVariableOp_2AssignVariableOpassignvariableop_2_fc_1_kernelIdentity_2:output:0*
_output_shapes
 *
dtype0N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:|
AssignVariableOp_3AssignVariableOpassignvariableop_3_fc_1_biasIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:А
AssignVariableOp_4AssignVariableOp assignvariableop_4_logits_kernelIdentity_4:output:0*
_output_shapes
 *
dtype0N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:~
AssignVariableOp_5AssignVariableOpassignvariableop_5_logits_biasIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
_output_shapes
:*
T0	Е
AssignVariableOp_6AssignVariableOp%assignvariableop_6_training_adam_iterIdentity_6:output:0*
_output_shapes
 *
dtype0	N

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:З
AssignVariableOp_7AssignVariableOp'assignvariableop_7_training_adam_beta_1Identity_7:output:0*
_output_shapes
 *
dtype0N

Identity_8IdentityRestoreV2:tensors:8*
_output_shapes
:*
T0З
AssignVariableOp_8AssignVariableOp'assignvariableop_8_training_adam_beta_2Identity_8:output:0*
_output_shapes
 *
dtype0N

Identity_9IdentityRestoreV2:tensors:9*
_output_shapes
:*
T0Ж
AssignVariableOp_9AssignVariableOp&assignvariableop_9_training_adam_decayIdentity_9:output:0*
_output_shapes
 *
dtype0P
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:С
AssignVariableOp_10AssignVariableOp/assignvariableop_10_training_adam_learning_rateIdentity_10:output:0*
dtype0*
_output_shapes
 P
Identity_11IdentityRestoreV2:tensors:11*
_output_shapes
:*
T0{
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0*
_output_shapes
 *
dtype0P
Identity_12IdentityRestoreV2:tensors:12*
_output_shapes
:*
T0{
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0*
_output_shapes
 *
dtype0P
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:С
AssignVariableOp_13AssignVariableOp/assignvariableop_13_training_adam_fc_0_kernel_mIdentity_13:output:0*
dtype0*
_output_shapes
 P
Identity_14IdentityRestoreV2:tensors:14*
_output_shapes
:*
T0П
AssignVariableOp_14AssignVariableOp-assignvariableop_14_training_adam_fc_0_bias_mIdentity_14:output:0*
_output_shapes
 *
dtype0P
Identity_15IdentityRestoreV2:tensors:15*
_output_shapes
:*
T0С
AssignVariableOp_15AssignVariableOp/assignvariableop_15_training_adam_fc_1_kernel_mIdentity_15:output:0*
dtype0*
_output_shapes
 P
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:П
AssignVariableOp_16AssignVariableOp-assignvariableop_16_training_adam_fc_1_bias_mIdentity_16:output:0*
dtype0*
_output_shapes
 P
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:У
AssignVariableOp_17AssignVariableOp1assignvariableop_17_training_adam_logits_kernel_mIdentity_17:output:0*
_output_shapes
 *
dtype0P
Identity_18IdentityRestoreV2:tensors:18*
_output_shapes
:*
T0С
AssignVariableOp_18AssignVariableOp/assignvariableop_18_training_adam_logits_bias_mIdentity_18:output:0*
_output_shapes
 *
dtype0P
Identity_19IdentityRestoreV2:tensors:19*
_output_shapes
:*
T0С
AssignVariableOp_19AssignVariableOp/assignvariableop_19_training_adam_fc_0_kernel_vIdentity_19:output:0*
_output_shapes
 *
dtype0P
Identity_20IdentityRestoreV2:tensors:20*
_output_shapes
:*
T0П
AssignVariableOp_20AssignVariableOp-assignvariableop_20_training_adam_fc_0_bias_vIdentity_20:output:0*
dtype0*
_output_shapes
 P
Identity_21IdentityRestoreV2:tensors:21*
_output_shapes
:*
T0С
AssignVariableOp_21AssignVariableOp/assignvariableop_21_training_adam_fc_1_kernel_vIdentity_21:output:0*
dtype0*
_output_shapes
 P
Identity_22IdentityRestoreV2:tensors:22*
_output_shapes
:*
T0П
AssignVariableOp_22AssignVariableOp-assignvariableop_22_training_adam_fc_1_bias_vIdentity_22:output:0*
_output_shapes
 *
dtype0P
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:У
AssignVariableOp_23AssignVariableOp1assignvariableop_23_training_adam_logits_kernel_vIdentity_23:output:0*
_output_shapes
 *
dtype0P
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:С
AssignVariableOp_24AssignVariableOp/assignvariableop_24_training_adam_logits_bias_vIdentity_24:output:0*
_output_shapes
 *
dtype0М
RestoreV2_1/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPHt
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B ╡
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
21
NoOpNoOp"/device:CPU:0*
_output_shapes
 ї
Identity_25Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: В
Identity_26IdentityIdentity_25:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: "#
identity_26Identity_26:output:0*y
_input_shapesh
f: :::::::::::::::::::::::::2(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112
RestoreV2_1RestoreV2_12*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192*
AssignVariableOp_24AssignVariableOp_242$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_4:+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : : : : : 
¤
т
>__inference_model_layer_call_and_return_conditional_losses_424
overcooked_observation,
(fc_0_statefulpartitionedcall_fc_0_kernel*
&fc_0_statefulpartitionedcall_fc_0_bias,
(fc_1_statefulpartitionedcall_fc_1_kernel*
&fc_1_statefulpartitionedcall_fc_1_bias0
,logits_statefulpartitionedcall_logits_kernel.
*logits_statefulpartitionedcall_logits_bias
identityИвfc_0/StatefulPartitionedCallвfc_1/StatefulPartitionedCallвlogits/StatefulPartitionedCallЙ
fc_0/StatefulPartitionedCallStatefulPartitionedCallovercooked_observation(fc_0_statefulpartitionedcall_fc_0_kernel&fc_0_statefulpartitionedcall_fc_0_bias*
Tin
2*
Tout
2*F
fAR?
=__inference_fc_0_layer_call_and_return_conditional_losses_345*'
_output_shapes
:         @**
config_proto

CPU

GPU 2J 8**
_gradient_op_typePartitionedCall-352Ш
fc_1/StatefulPartitionedCallStatefulPartitionedCall%fc_0/StatefulPartitionedCall:output:0(fc_1_statefulpartitionedcall_fc_1_kernel&fc_1_statefulpartitionedcall_fc_1_bias**
_gradient_op_typePartitionedCall-382*'
_output_shapes
:         @**
config_proto

CPU

GPU 2J 8*
Tin
2*F
fAR?
=__inference_fc_1_layer_call_and_return_conditional_losses_375*
Tout
2д
logits/StatefulPartitionedCallStatefulPartitionedCall%fc_1/StatefulPartitionedCall:output:0,logits_statefulpartitionedcall_logits_kernel*logits_statefulpartitionedcall_logits_bias*
Tin
2**
_gradient_op_typePartitionedCall-411*'
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*
Tout
2*H
fCRA
?__inference_logits_layer_call_and_return_conditional_losses_404╬
IdentityIdentity'logits/StatefulPartitionedCall:output:0^fc_0/StatefulPartitionedCall^fc_1/StatefulPartitionedCall^logits/StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*>
_input_shapes-
+:         `::::::2<
fc_0/StatefulPartitionedCallfc_0/StatefulPartitionedCall2<
fc_1/StatefulPartitionedCallfc_1/StatefulPartitionedCall2@
logits/StatefulPartitionedCalllogits/StatefulPartitionedCall:6 2
0
_user_specified_nameOvercooked_observation: : : : : : 
╘
╫
__inference__wrapped_model_327
overcooked_observation0
,model_fc_0_matmul_readvariableop_fc_0_kernel/
+model_fc_0_biasadd_readvariableop_fc_0_bias0
,model_fc_1_matmul_readvariableop_fc_1_kernel/
+model_fc_1_biasadd_readvariableop_fc_1_bias4
0model_logits_matmul_readvariableop_logits_kernel3
/model_logits_biasadd_readvariableop_logits_bias
identityИв!model/fc_0/BiasAdd/ReadVariableOpв model/fc_0/MatMul/ReadVariableOpв!model/fc_1/BiasAdd/ReadVariableOpв model/fc_1/MatMul/ReadVariableOpв#model/logits/BiasAdd/ReadVariableOpв"model/logits/MatMul/ReadVariableOpН
 model/fc_0/MatMul/ReadVariableOpReadVariableOp,model_fc_0_matmul_readvariableop_fc_0_kernel*
_output_shapes

:`@*
dtype0П
model/fc_0/MatMulMatMulovercooked_observation(model/fc_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Й
!model/fc_0/BiasAdd/ReadVariableOpReadVariableOp+model_fc_0_biasadd_readvariableop_fc_0_bias*
_output_shapes
:@*
dtype0Ч
model/fc_0/BiasAddBiasAddmodel/fc_0/MatMul:product:0)model/fc_0/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:         @*
T0f
model/fc_0/ReluRelumodel/fc_0/BiasAdd:output:0*'
_output_shapes
:         @*
T0Н
 model/fc_1/MatMul/ReadVariableOpReadVariableOp,model_fc_1_matmul_readvariableop_fc_1_kernel*
dtype0*
_output_shapes

:@@Ц
model/fc_1/MatMulMatMulmodel/fc_0/Relu:activations:0(model/fc_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Й
!model/fc_1/BiasAdd/ReadVariableOpReadVariableOp+model_fc_1_biasadd_readvariableop_fc_1_bias*
dtype0*
_output_shapes
:@Ч
model/fc_1/BiasAddBiasAddmodel/fc_1/MatMul:product:0)model/fc_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @f
model/fc_1/ReluRelumodel/fc_1/BiasAdd:output:0*
T0*'
_output_shapes
:         @У
"model/logits/MatMul/ReadVariableOpReadVariableOp0model_logits_matmul_readvariableop_logits_kernel*
_output_shapes

:@*
dtype0Ъ
model/logits/MatMulMatMulmodel/fc_1/Relu:activations:0*model/logits/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         П
#model/logits/BiasAdd/ReadVariableOpReadVariableOp/model_logits_biasadd_readvariableop_logits_bias*
dtype0*
_output_shapes
:Э
model/logits/BiasAddBiasAddmodel/logits/MatMul:product:0+model/logits/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ╛
IdentityIdentitymodel/logits/BiasAdd:output:0"^model/fc_0/BiasAdd/ReadVariableOp!^model/fc_0/MatMul/ReadVariableOp"^model/fc_1/BiasAdd/ReadVariableOp!^model/fc_1/MatMul/ReadVariableOp$^model/logits/BiasAdd/ReadVariableOp#^model/logits/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         "
identityIdentity:output:0*>
_input_shapes-
+:         `::::::2H
"model/logits/MatMul/ReadVariableOp"model/logits/MatMul/ReadVariableOp2D
 model/fc_0/MatMul/ReadVariableOp model/fc_0/MatMul/ReadVariableOp2F
!model/fc_1/BiasAdd/ReadVariableOp!model/fc_1/BiasAdd/ReadVariableOp2F
!model/fc_0/BiasAdd/ReadVariableOp!model/fc_0/BiasAdd/ReadVariableOp2J
#model/logits/BiasAdd/ReadVariableOp#model/logits/BiasAdd/ReadVariableOp2D
 model/fc_1/MatMul/ReadVariableOp model/fc_1/MatMul/ReadVariableOp: : :6 2
0
_user_specified_nameOvercooked_observation: : : : 
э	
р
#__inference_model_layer_call_fn_494
overcooked_observation'
#statefulpartitionedcall_fc_0_kernel%
!statefulpartitionedcall_fc_0_bias'
#statefulpartitionedcall_fc_1_kernel%
!statefulpartitionedcall_fc_1_bias)
%statefulpartitionedcall_logits_kernel'
#statefulpartitionedcall_logits_bias
identityИвStatefulPartitionedCallУ
StatefulPartitionedCallStatefulPartitionedCallovercooked_observation#statefulpartitionedcall_fc_0_kernel!statefulpartitionedcall_fc_0_bias#statefulpartitionedcall_fc_1_kernel!statefulpartitionedcall_fc_1_bias%statefulpartitionedcall_logits_kernel#statefulpartitionedcall_logits_bias*
Tout
2**
config_proto

CPU

GPU 2J 8*G
fBR@
>__inference_model_layer_call_and_return_conditional_losses_484*
Tin
	2**
_gradient_op_typePartitionedCall-485*'
_output_shapes
:         В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*>
_input_shapes-
+:         `::::::22
StatefulPartitionedCallStatefulPartitionedCall:6 2
0
_user_specified_nameOvercooked_observation: : : : : : 
є
┌
=__inference_fc_1_layer_call_and_return_conditional_losses_609

inputs%
!matmul_readvariableop_fc_1_kernel$
 biasadd_readvariableop_fc_1_bias
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpw
MatMul/ReadVariableOpReadVariableOp!matmul_readvariableop_fc_1_kernel*
dtype0*
_output_shapes

:@@i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:         @*
T0s
BiasAdd/ReadVariableOpReadVariableOp biasadd_readvariableop_fc_1_bias*
dtype0*
_output_shapes
:@v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:         @*
T0P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         @Л
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*.
_input_shapes
:         @::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
═
╥
>__inference_model_layer_call_and_return_conditional_losses_456

inputs,
(fc_0_statefulpartitionedcall_fc_0_kernel*
&fc_0_statefulpartitionedcall_fc_0_bias,
(fc_1_statefulpartitionedcall_fc_1_kernel*
&fc_1_statefulpartitionedcall_fc_1_bias0
,logits_statefulpartitionedcall_logits_kernel.
*logits_statefulpartitionedcall_logits_bias
identityИвfc_0/StatefulPartitionedCallвfc_1/StatefulPartitionedCallвlogits/StatefulPartitionedCall∙
fc_0/StatefulPartitionedCallStatefulPartitionedCallinputs(fc_0_statefulpartitionedcall_fc_0_kernel&fc_0_statefulpartitionedcall_fc_0_bias**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:         @*
Tin
2*F
fAR?
=__inference_fc_0_layer_call_and_return_conditional_losses_345**
_gradient_op_typePartitionedCall-352*
Tout
2Ш
fc_1/StatefulPartitionedCallStatefulPartitionedCall%fc_0/StatefulPartitionedCall:output:0(fc_1_statefulpartitionedcall_fc_1_kernel&fc_1_statefulpartitionedcall_fc_1_bias*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2**
_gradient_op_typePartitionedCall-382*F
fAR?
=__inference_fc_1_layer_call_and_return_conditional_losses_375*'
_output_shapes
:         @д
logits/StatefulPartitionedCallStatefulPartitionedCall%fc_1/StatefulPartitionedCall:output:0,logits_statefulpartitionedcall_logits_kernel*logits_statefulpartitionedcall_logits_bias*'
_output_shapes
:         **
_gradient_op_typePartitionedCall-411*
Tout
2*H
fCRA
?__inference_logits_layer_call_and_return_conditional_losses_404**
config_proto

CPU

GPU 2J 8*
Tin
2╬
IdentityIdentity'logits/StatefulPartitionedCall:output:0^fc_0/StatefulPartitionedCall^fc_1/StatefulPartitionedCall^logits/StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*>
_input_shapes-
+:         `::::::2<
fc_1/StatefulPartitionedCallfc_1/StatefulPartitionedCall2@
logits/StatefulPartitionedCalllogits/StatefulPartitionedCall2<
fc_0/StatefulPartitionedCallfc_0/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : 
є
┌
=__inference_fc_0_layer_call_and_return_conditional_losses_591

inputs%
!matmul_readvariableop_fc_0_kernel$
 biasadd_readvariableop_fc_0_bias
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpw
MatMul/ReadVariableOpReadVariableOp!matmul_readvariableop_fc_0_kernel*
_output_shapes

:`@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:         @*
T0s
BiasAdd/ReadVariableOpReadVariableOp biasadd_readvariableop_fc_0_bias*
dtype0*
_output_shapes
:@v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         @Л
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*.
_input_shapes
:         `::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
й
р
?__inference_logits_layer_call_and_return_conditional_losses_404

inputs'
#matmul_readvariableop_logits_kernel&
"biasadd_readvariableop_logits_bias
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpy
MatMul/ReadVariableOpReadVariableOp#matmul_readvariableop_logits_kernel*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:         *
T0u
BiasAdd/ReadVariableOpReadVariableOp"biasadd_readvariableop_logits_bias*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:         *
T0Й
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:         *
T0"
identityIdentity:output:0*.
_input_shapes
:         @::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
┴
Я
>__inference_model_layer_call_and_return_conditional_losses_558

inputs*
&fc_0_matmul_readvariableop_fc_0_kernel)
%fc_0_biasadd_readvariableop_fc_0_bias*
&fc_1_matmul_readvariableop_fc_1_kernel)
%fc_1_biasadd_readvariableop_fc_1_bias.
*logits_matmul_readvariableop_logits_kernel-
)logits_biasadd_readvariableop_logits_bias
identityИвfc_0/BiasAdd/ReadVariableOpвfc_0/MatMul/ReadVariableOpвfc_1/BiasAdd/ReadVariableOpвfc_1/MatMul/ReadVariableOpвlogits/BiasAdd/ReadVariableOpвlogits/MatMul/ReadVariableOpБ
fc_0/MatMul/ReadVariableOpReadVariableOp&fc_0_matmul_readvariableop_fc_0_kernel*
_output_shapes

:`@*
dtype0s
fc_0/MatMulMatMulinputs"fc_0/MatMul/ReadVariableOp:value:0*'
_output_shapes
:         @*
T0}
fc_0/BiasAdd/ReadVariableOpReadVariableOp%fc_0_biasadd_readvariableop_fc_0_bias*
dtype0*
_output_shapes
:@Е
fc_0/BiasAddBiasAddfc_0/MatMul:product:0#fc_0/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:         @*
T0Z
	fc_0/ReluRelufc_0/BiasAdd:output:0*
T0*'
_output_shapes
:         @Б
fc_1/MatMul/ReadVariableOpReadVariableOp&fc_1_matmul_readvariableop_fc_1_kernel*
dtype0*
_output_shapes

:@@Д
fc_1/MatMulMatMulfc_0/Relu:activations:0"fc_1/MatMul/ReadVariableOp:value:0*'
_output_shapes
:         @*
T0}
fc_1/BiasAdd/ReadVariableOpReadVariableOp%fc_1_biasadd_readvariableop_fc_1_bias*
dtype0*
_output_shapes
:@Е
fc_1/BiasAddBiasAddfc_1/MatMul:product:0#fc_1/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:         @*
T0Z
	fc_1/ReluRelufc_1/BiasAdd:output:0*'
_output_shapes
:         @*
T0З
logits/MatMul/ReadVariableOpReadVariableOp*logits_matmul_readvariableop_logits_kernel*
dtype0*
_output_shapes

:@И
logits/MatMulMatMulfc_1/Relu:activations:0$logits/MatMul/ReadVariableOp:value:0*'
_output_shapes
:         *
T0Г
logits/BiasAdd/ReadVariableOpReadVariableOp)logits_biasadd_readvariableop_logits_bias*
dtype0*
_output_shapes
:Л
logits/BiasAddBiasAddlogits/MatMul:product:0%logits/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ф
IdentityIdentitylogits/BiasAdd:output:0^fc_0/BiasAdd/ReadVariableOp^fc_0/MatMul/ReadVariableOp^fc_1/BiasAdd/ReadVariableOp^fc_1/MatMul/ReadVariableOp^logits/BiasAdd/ReadVariableOp^logits/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         "
identityIdentity:output:0*>
_input_shapes-
+:         `::::::28
fc_0/MatMul/ReadVariableOpfc_0/MatMul/ReadVariableOp2:
fc_1/BiasAdd/ReadVariableOpfc_1/BiasAdd/ReadVariableOp2:
fc_0/BiasAdd/ReadVariableOpfc_0/BiasAdd/ReadVariableOp2>
logits/BiasAdd/ReadVariableOplogits/BiasAdd/ReadVariableOp28
fc_1/MatMul/ReadVariableOpfc_1/MatMul/ReadVariableOp2<
logits/MatMul/ReadVariableOplogits/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : : : "ЖL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*╟
serving_default│
Y
Overcooked_observation?
(serving_default_Overcooked_observation:0         `:
logits0
StatefulPartitionedCall:0         tensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:лЧ
■&
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
X__call__"▒$
_tf_keras_modelЧ${"class_name": "Model", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": null, "batch_input_shape": null, "config": {"name": "model", "layers": [{"name": "Overcooked_observation", "class_name": "InputLayer", "config": {"batch_input_shape": [null, 96], "dtype": "float32", "sparse": false, "ragged": false, "name": "Overcooked_observation"}, "inbound_nodes": []}, {"name": "fc_0", "class_name": "Dense", "config": {"name": "fc_0", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["Overcooked_observation", 0, 0, {}]]]}, {"name": "fc_1", "class_name": "Dense", "config": {"name": "fc_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["fc_0", 0, 0, {}]]]}, {"name": "logits", "class_name": "Dense", "config": {"name": "logits", "trainable": true, "dtype": "float32", "units": 6, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["fc_1", 0, 0, {}]]]}], "input_layers": [["Overcooked_observation", 0, 0]], "output_layers": [["logits", 0, 0]]}, "input_spec": null, "activity_regularizer": null, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model", "layers": [{"name": "Overcooked_observation", "class_name": "InputLayer", "config": {"batch_input_shape": [null, 96], "dtype": "float32", "sparse": false, "ragged": false, "name": "Overcooked_observation"}, "inbound_nodes": []}, {"name": "fc_0", "class_name": "Dense", "config": {"name": "fc_0", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["Overcooked_observation", 0, 0, {}]]]}, {"name": "fc_1", "class_name": "Dense", "config": {"name": "fc_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["fc_0", 0, 0, {}]]]}, {"name": "logits", "class_name": "Dense", "config": {"name": "logits", "trainable": true, "dtype": "float32", "units": 6, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["fc_1", 0, 0, {}]]]}], "input_layers": [["Overcooked_observation", 0, 0]], "output_layers": [["logits", 0, 0]]}}, "training_config": {"loss": {"class_name": "SparseCategoricalCrossentropy", "config": {"reduction": "auto", "name": "sparse_categorical_crossentropy", "from_logits": true}}, "metrics": ["sparse_categorical_accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
Д
	variables
trainable_variables
regularization_losses
	keras_api
*Y&call_and_return_all_conditional_losses
Z__call__"ї
_tf_keras_layer█{"class_name": "InputLayer", "name": "Overcooked_observation", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 96], "config": {"batch_input_shape": [null, 96], "dtype": "float32", "sparse": false, "ragged": false, "name": "Overcooked_observation"}, "input_spec": null, "activity_regularizer": null}
┼

kernel
bias
_callable_losses
	variables
trainable_variables
regularization_losses
	keras_api
*[&call_and_return_all_conditional_losses
\__call__"К
_tf_keras_layerЁ{"class_name": "Dense", "name": "fc_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "fc_0", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 96}}}, "activity_regularizer": null}
┼

kernel
bias
_callable_losses
	variables
trainable_variables
regularization_losses
	keras_api
*]&call_and_return_all_conditional_losses
^__call__"К
_tf_keras_layerЁ{"class_name": "Dense", "name": "fc_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "fc_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "activity_regularizer": null}
╩

kernel
bias
_callable_losses
 	variables
!trainable_variables
"regularization_losses
#	keras_api
*_&call_and_return_all_conditional_losses
`__call__"П
_tf_keras_layerї{"class_name": "Dense", "name": "logits", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "logits", "trainable": true, "dtype": "float32", "units": 6, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "activity_regularizer": null}
┐
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
╖
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
Ъ
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
:`@2fc_0/kernel
:@2	fc_0/bias
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
Ъ
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
:@@2fc_1/kernel
:@2	fc_1/bias
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
Ъ
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
:@2logits/kernel
:2logits/bias
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
Ъ
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
:	 (2training/Adam/iter
: (2training/Adam/beta_1
: (2training/Adam/beta_2
: (2training/Adam/decay
%:# (2training/Adam/learning_rate
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
А
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
c__call__"╜
_tf_keras_layerг{"class_name": "MeanMetricWrapper", "name": "sparse_categorical_accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sparse_categorical_accuracy", "dtype": "float32"}, "input_spec": null, "activity_regularizer": null}
:  (2total
:  (2count
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
Ъ
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
+:)`@2training/Adam/fc_0/kernel/m
%:#@2training/Adam/fc_0/bias/m
+:)@@2training/Adam/fc_1/kernel/m
%:#@2training/Adam/fc_1/bias/m
-:+@2training/Adam/logits/kernel/m
':%2training/Adam/logits/bias/m
+:)`@2training/Adam/fc_0/kernel/v
%:#@2training/Adam/fc_0/bias/v
+:)@@2training/Adam/fc_1/kernel/v
%:#@2training/Adam/fc_1/bias/v
-:+@2training/Adam/logits/kernel/v
':%2training/Adam/logits/bias/v
ы2ш
__inference__wrapped_model_327┼
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *5в2
0К-
Overcooked_observation         `
╞2├
>__inference_model_layer_call_and_return_conditional_losses_558
>__inference_model_layer_call_and_return_conditional_losses_440
>__inference_model_layer_call_and_return_conditional_losses_424
>__inference_model_layer_call_and_return_conditional_losses_534└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
┌2╫
#__inference_model_layer_call_fn_494
#__inference_model_layer_call_fn_580
#__inference_model_layer_call_fn_466
#__inference_model_layer_call_fn_569└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╠2╔╞
╜▓╣
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
╠2╔╞
╜▓╣
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
ч2ф
=__inference_fc_0_layer_call_and_return_conditional_losses_591в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╠2╔
"__inference_fc_0_layer_call_fn_598в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ч2ф
=__inference_fc_1_layer_call_and_return_conditional_losses_609в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╠2╔
"__inference_fc_1_layer_call_fn_616в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
щ2ц
?__inference_logits_layer_call_and_return_conditional_losses_626в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╬2╦
$__inference_logits_layer_call_fn_633в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
?B=
!__inference_signature_wrapper_507Overcooked_observation
╠2╔╞
╜▓╣
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
╠2╔╞
╜▓╣
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 ║
!__inference_signature_wrapper_507ФYвV
в 
OкL
J
Overcooked_observation0К-
Overcooked_observation         `"/к,
*
logits К
logits         В
#__inference_model_layer_call_fn_580[7в4
-в*
 К
inputs         `
p 

 
к "К         к
>__inference_model_layer_call_and_return_conditional_losses_534h7в4
-в*
 К
inputs         `
p

 
к "%в"
К
0         
Ъ В
#__inference_model_layer_call_fn_569[7в4
-в*
 К
inputs         `
p

 
к "К         u
"__inference_fc_0_layer_call_fn_598O/в,
%в"
 К
inputs         `
к "К         @Э
=__inference_fc_0_layer_call_and_return_conditional_losses_591\/в,
%в"
 К
inputs         `
к "%в"
К
0         @
Ъ Ь
__inference__wrapped_model_327z?в<
5в2
0К-
Overcooked_observation         `
к "/к,
*
logits К
logits         Я
?__inference_logits_layer_call_and_return_conditional_losses_626\/в,
%в"
 К
inputs         @
к "%в"
К
0         
Ъ w
$__inference_logits_layer_call_fn_633O/в,
%в"
 К
inputs         @
к "К         Т
#__inference_model_layer_call_fn_466kGвD
=в:
0К-
Overcooked_observation         `
p

 
к "К         Т
#__inference_model_layer_call_fn_494kGвD
=в:
0К-
Overcooked_observation         `
p 

 
к "К         к
>__inference_model_layer_call_and_return_conditional_losses_558h7в4
-в*
 К
inputs         `
p 

 
к "%в"
К
0         
Ъ Э
=__inference_fc_1_layer_call_and_return_conditional_losses_609\/в,
%в"
 К
inputs         @
к "%в"
К
0         @
Ъ u
"__inference_fc_1_layer_call_fn_616O/в,
%в"
 К
inputs         @
к "К         @║
>__inference_model_layer_call_and_return_conditional_losses_424xGвD
=в:
0К-
Overcooked_observation         `
p

 
к "%в"
К
0         
Ъ ║
>__inference_model_layer_call_and_return_conditional_losses_440xGвD
=в:
0К-
Overcooked_observation         `
p 

 
к "%в"
К
0         
Ъ 