# psenet_cpp

## The model is trained by liuheng92/tensorflow_PSENet.
## Here is my code to convert ckpt to pb:

```
def freeze(ckpt_path=None, save_path=None):

    from tensorflow.python.tools import freeze_graph  # , optimize_for_inference_lib

    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        seg_maps_pred = model.model(input_images, is_training=False)

        tf.identity(seg_maps_pred, name='seg_maps')

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        with tf.Session() as sess:
            saver.restore(sess, ckpt_path)

            print('Freeze Model Will Saved at ', save_path)
            fdir, name = os.path.split(save_path)
            tf.train.write_graph(sess.graph_def, fdir, name, as_text=True)

            freeze_graph.freeze_graph(
                input_graph=save_path,
                input_saver='',
                input_binary=False,
                input_checkpoint=ckpt_path,
                output_node_names='seg_maps',
                restore_op_name='',
                filename_tensor_name='',
                output_graph=save_path,
                clear_devices=True,
                initializer_nodes='',
            )
```
