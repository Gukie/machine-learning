import tensorflow as tf

node1 = tf.constant(3.0,dtype=tf.float32)
node2 = tf.constant(4.0)
print(node1,node2)

print('####session test start#####')
session = tf.Session();
print(session.run([node1,node2]))

# create new node by other nodes
print('## create new node')
node3 = tf.add(node1,node2)
print("node3:",node3)
print("session.run([node3]):",session.run([node3]))

# use placeholder rather than constant
print('## use placeholder rather than constant')
ph1 = tf.placeholder(tf.float32)
ph2 = tf.placeholder(tf.float32)
sum = ph1 + ph2
print(sum)
print(session.run(sum, {ph1:32, ph2:34}))

# compose complicated computation
multple = sum*23
print(session.run(multple, {ph1:2, ph2:3}))

# use variables
var1 = tf.Variable([.3],dtype=tf.float32)
var2 = tf.Variable([-.3],dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = var1 * x + var2
# following 2 line is needed to init variables, otherwise exception will thrown
init = tf.global_variables_initializer()
session.run(init)
print(session.run(linear_model, {x:[1, 2, 3, 4]}))

# to evaluate how good the model is
y = tf.placeholder(dtype=tf.float32)
delta_square = tf.square(linear_model-y);
loss = tf.reduce_sum(delta_square)
print(session.run(loss,{x:[1,2,3,4],y:[0,-1,-2,-3]}))

# assgin the variables after initializing
fixVar1 = tf.assign(var1,[-1.])
fixVar2 = tf.assign(var2,[1.])
session.run([fixVar1,fixVar2])
print(session.run(loss,{x:[1,2,3,4],y:[0,-1,-2,-3]}))