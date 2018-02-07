import tensorflow as tf

class MNet:
    def __init__(self):
        self.graph=tf.Graph()
        self.sess=tf.Session(self.graph)
        with self.graph.as_default():
            self._X=tf.placeholder(tf.float32,[None,32,32,3])
            self._Y=tf.placeholder(tf.float32,[None,2])

            self._w={
            'wc1_1':tf.get_variable('wc1_1',[3,3,3,16],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
            'wc1_2':tf.get_variable('wc1_2',[3,3,16,16],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
            'wc2_1':tf.get_variable('wc2_1',[3,3,16,32],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
            'wc2_2':tf.get_variable('wc2_2',[3,3,32,32],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
            'wc3_1':tf.get_variable('wc3_1',[3,3,32,64],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
            'wc3_2':tf.get_variable('wc3_2',[3,3,64,64],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
            'fc1':tf.get_variable('fc1',[4,4,64,128],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
            'fc2':tf.get_variable('fc2',[1,1,128,5],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
        }

            self._b={
            'wb1_1':tf.get_variable('wb1_1',[16],initializer=tf.constant_initializer(0.)),
            'wb1_2':tf.get_variable('wb1_2',[16],initializer=tf.constant_initializer(0.)),
            'wb2_1':tf.get_variable('wb2_1',[32],initializer=tf.constant_initializer(0.)),
            'wb2_2':tf.get_variable('wb2_2',[32],initializer=tf.constant_initializer(0.)),
            'wb3_1':tf.get_variable('wb3_1',[64],initializer=tf.constant_initializer(0.)),
            'wb3_2':tf.get_variable('wb3_2',[64],initializer=tf.constant_initializer(0.)),
            'fb1':tf.get_variable('fb1',[128],initializer=tf.constant_initializer(0.)),
            'fb2':tf.get_variable('fb2',[5],initializer=tf.constant_initializer(0.)),
         }

            conv1_1=tf.nn.conv2d(self._X,self._w['wc1_1'],[1,1,1,1],padding='SAME')
            conv1_1=tf.nn.relu(tf.nn.bias_add(conv1_1,self._b['wb1_1']))
            conv1_2=tf.nn.conv2d(conv1_1,self._w['wc1_2'],[1,1,1,1],padding='SAME')
            conv1_2=tf.nn.relu(tf.nn.bias_add(conv1_2,self._b['wb1_2']))
            conv1_2=tf.nn.max_pool(conv1_2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

            conv2_1=tf.nn.conv2d(conv1_2,self._w['wc2_1'],[1,1,1,1],padding='SAME')
            conv2_1=tf.nn.relu(tf.nn.bias_add(conv2_1,self._b['wb2_1']))
            conv2_2=tf.nn.conv2d(conv2_1,self._w['wc2_2'],[1,1,1,1],padding='SAME')
            conv2_2=tf.nn.relu(tf.nn.bias_add(conv2_2,self._b['wb2_2']))
            conv2_2=tf.nn.max_pool(conv2_2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

            conv3_1=tf.nn.conv2d(conv2_2,self._w['wc3_1'],[1,1,1,1],padding='SAME')
            conv3_1=tf.nn.relu(tf.nn.bias_add(conv3_1,self._b['wb3_1']))
            conv3_2=tf.nn.conv2d(conv3_1,self._w['wc3_2'],[1,1,1,1],padding='SAME')
            conv3_2=tf.nn.relu(tf.nn.bias_add(conv3_2,self._b['wb3_2']))
            conv3_2=tf.nn.max_pool(conv3_2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

            fc1=tf.nn.conv2d(conv3_2,self._w['fc1'],[1,1,1,1],padding='VALID')
            fc1=tf.nn.relu(tf.nn.bias_add(fc1,self._b['fb1']))
            score=tf.nn.bias_add(tf.nn.conv2d(fc1,self._w['fc2'],[1,1,1,1],padding='VALID'),self._b['fb2'])

            self.logit=tf.reshape(score,[-1,score.get_shape().as_list()[3]])

            self.loss=tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logit,labels=self._Y)

            variable_to_train=[]

            self.avg_loss=tf.reduce_mean(self.loss)

            self.opt=tf.train.AdamOptimizer(0.001)

            self.train_op=self.opt.minimize(self.loss)

            correct_pred=tf.equal(tf.cast(tf.arg_max(self.logit,1),tf.int32),tf.cast(tf.arg_max(self._Y,1),tf.int32))

            self.tp=tf.reduce_sum(tf.)


            self.accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

            init=tf.global_variables_initializer()


    def dictXY(self,X,Y):
        return {self._X:X,self._Y:Y}

    def dictX(self,X):
       return {self._X:X}