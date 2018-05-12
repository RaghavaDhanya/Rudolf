import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

# def load_graph(model_file):
#     graph = tf.Graph()
#     graph_def = tf.GraphDef()

#     with open(model_file, "rb") as f:
#         graph_def.ParseFromString(f.read())
#     with graph.as_default():
#         tf.import_graph_def(graph_def)

#     input_name = graph.get_operations()[0].name+':0'
#     output_name = graph.get_operations()[-1].name+':0'
#     return graph,input_name,output_name


# gen_graph,gen_input_name,gen_output_name=load_graph('generator.pb')
# print("Graph loaded, input={},output={}".format(gen_input_name,gen_output_name))

# dis_graph,dis_input_name,dis_output_name=load_graph('discriminator.pb')
# print("Graph loaded, input={},output={}".format(dis_input_name,dis_output_name))



# z = gen_graph.get_tensor_by_name(gen_input_name)
# imgs = gen_graph.get_tensor_by_name(gen_output_name) 

# print(z.shape,imgs.shape)

# imgs = dis_graph.get_tensor_by_name(dis_input_name)
# valid = dis_graph.get_tensor_by_name(dis_output_name) 
# print(imgs.shape,valid.shape)



gen_graph_def = tf.GraphDef()
with open('generator.pb', "rb") as f:
    gen_graph_def.ParseFromString(f.read())

gen_input_name="input_2:0"
gen_output_name="output_node0:0"
z=tf.placeholder(tf.float32,shape=(1,100))
gen_imgs,=tf.import_graph_def(gen_graph_def,input_map={gen_input_name:z}, return_elements=[gen_output_name])
print("loaded generator")

dis_graph_def = tf.GraphDef()
with open('discriminator.pb', "rb") as f:
    dis_graph_def.ParseFromString(f.read())
dis_imgs=tf.reshape(gen_imgs,shape=[1,28,28,1])
print(dis_imgs.shape)
dis_input_name="input_1:0"
dis_output_name="output_node0:0"

valid,=tf.import_graph_def(dis_graph_def,input_map={dis_input_name:dis_imgs}, return_elements=[dis_output_name])
print("loaded discriminator")

mask = tf.placeholder(tf.float32, (28,28,1), name='mask')
image = tf.placeholder(tf.float32, (28,28,1), name='image')
contextual_loss = tf.reduce_sum(
    tf.contrib.layers.flatten(
        tf.abs(tf.multiply(mask, gen_imgs) - tf.multiply(mask, image))), 1)
perceptual_loss = tf.log(1-valid)
complete_loss = contextual_loss + 0.1*perceptual_loss
grad_complete_loss = tf.gradients(complete_loss, z)



with tf.Session() as sess:
    scale = 0.3
    mask_val=np.ones((28,28,1))
    l = int(28*scale)
    u = int(28*(1.0-scale))
    mask_val[l:u, l:u, :] = 0.0
    in_image=(cv2.imread('img_54.jpg',0)).reshape(28,28,1)
    in_image=(in_image.astype(np.float32)-127.5)/127.5
    zhats=np.random.normal(0, 1, (1, 100))
    momentum,lr=0.9,0.01
    v=0
    for i in range(1000):
        fd={
        mask:mask_val,
        image:in_image,
        z:zhats
        }
        outputs=[complete_loss, grad_complete_loss, gen_imgs]
        loss,grad,g_imgs=sess.run(outputs,feed_dict=fd)
        v_prev = np.copy(v)
        v = momentum*v - lr*grad[0]
        zhats += -momentum * v_prev + (1+momentum)*v
        zhats = np.clip(zhats, 0, 1)
        if(i%10==0):
            print("iter{}".format(i))

    g_imgs=sess.run(gen_imgs,feed_dict={z:zhats})

    completed_img=np.multiply(g_imgs,1-mask_val)+np.multiply(in_image,mask_val)
    completed_img=(1+completed_img)/2

    original_image=(1+in_image)/2
    masked_image=np.multiply(in_image,mask_val)

    fig, axs = plt.subplots(1, 3)
    axs[0].set_title("Original")
    axs[0].imshow(original_image[:,:,0],cmap='gray')
    axs[0].axis('off')
    axs[1].set_title("Masked")
    axs[1].imshow(masked_image[:,:,0],cmap='gray')
    axs[1].axis('off')
    axs[2].set_title("Completed")
    axs[2].imshow(completed_img[:,:,0],cmap='gray')
    axs[2].axis('off')
    plt.show()
    # plt.imshow(completed_img[:,:,0],cmap='gray')
    # plt.show()








# with tf.Session() as sess:
#     writer = tf.summary.FileWriter('logs', sess.graph)
#     img,v=sess.run([gen_imgs,valid],feed_dict={z:np.random.normal(0, 1, (1, 100))})
#     writer.close()
#     print(v)
#     img=(1+img)/2
#     plt.imshow(img[:,:,0],cmap='gray')
#     plt.show()