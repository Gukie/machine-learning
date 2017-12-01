import tensorflow as tf

node1 = tf.constant(3.0,dtype=tf.float32)
node2 = tf.constant(4.0)
print(node1,node2)

print('####session test start#####')
session = tf.Session();
print(session.run([node1,node2]))


print('## create new node')
node3 = tf.add(node1,node2)
print("node3:",node3)
print("session.run([node3]):",session.run([node3]))