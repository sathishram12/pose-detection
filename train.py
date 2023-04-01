import datetime
import tensorflow as tf
import nets.mobilenet_model as model

from datetime import timedelta
from timeit import default_timer as timer
from config import config
#from data_generator import DataGenerator
from dataset.generators import get_dataset
from dataset.label_maps import PredictionData
from tensorflow.keras.optimizers import Adam
from util import plot_to_image, probe_model

lr = 2.5e-5
checkpoints_folder = './tf_ckpts_patient_monitoring'
output_weights = 'output/patient_monitoring'

ds_train, ds_train_size = get_dataset(config.TRAIN_ANNO_FILE, config.TRAIN_IMG_DIR, config.BATCH_SIZE,False, config.IMAGE_SHAPE)
ds_val, ds_val_size = get_dataset(config.VAL_ANNO_FILE, config.VAL_IMG_DIR, config.BATCH_SIZE, True, config.IMAGE_SHAPE)
print(f"Training samples: {ds_train_size} , Validation samples: {ds_val_size}")
steps_per_epoch = ds_train_size // config.BATCH_SIZE
steps_per_epoch_val = ds_val_size // config.BATCH_SIZE

#ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
#ds_val = ds_val.prefetch(tf.data.AUTOTUNE)

def kp_map_loss(kp_maps_true,kp_maps_pred):
    loss = tf.keras.backend.binary_crossentropy(kp_maps_true,kp_maps_pred)
    loss = tf.reduce_mean(loss)*config.LOSS_WEIGHTS['heatmap']
    return loss

def short_offset_loss(short_offset_true, short_offsets_pred, kp_maps_true):
    loss = tf.abs(short_offset_true-short_offsets_pred)
    loss = loss*tf.concat([kp_maps_true, kp_maps_true], 3) #tf_repeat(kp_maps_true,[1,1,1,2])
    loss = tf.reduce_sum(loss) / (tf.reduce_sum(kp_maps_true))
    return loss*config.LOSS_WEIGHTS['short']

def mid_offset_loss(mid_offset_true,mid_offset_pred,kp_maps_true):
    loss = tf.abs(mid_offset_pred-mid_offset_true)
    recorded_maps = []
    for mid_idx, edge in enumerate(config.EDGES + [edge[::-1] for edge in config.EDGES]):
        from_kp = edge[0]
        recorded_maps.extend([kp_maps_true[:,:,:,from_kp], kp_maps_true[:,:,:,from_kp]])
    recorded_maps = tf.stack(recorded_maps,axis=-1)
    # print(recorded_maps)
    loss = loss*recorded_maps
    loss = tf.reduce_sum(loss)/(tf.reduce_sum(recorded_maps))
    return loss*config.LOSS_WEIGHTS['mid']

def get_losses(y_true, y_pred):
    losses = []
    losses.append(kp_map_loss(y_true[0],y_pred[0]))
    losses.append(short_offset_loss(y_true[1], y_pred[1], y_true[0]))
    losses.append(mid_offset_loss(y_true[2], y_pred[2],y_true[0]))
    return losses

@tf.function
def train_one_step(model, optimizer, x, y_true):
    
    with tf.GradientTape() as tape:
        y_pred = model(x)
        losses = get_losses(y_true, y_pred)
        total_loss = tf.reduce_sum(losses)

    grads = tape.gradient(total_loss, model.trainable_variables)
    #optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return grads,losses, total_loss

def train(ds_train, ds_val, model, optimizer, ckpt, last_epoch, last_step, max_epochs, steps_per_epoch):
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_loss_kpmaps = tf.keras.metrics.Mean('train_loss_kpmaps', dtype=tf.float32)
    train_loss_short_offset = tf.keras.metrics.Mean('train_loss_short_offset', dtype=tf.float32)
    train_loss_mid_offset = tf.keras.metrics.Mean('train_loss_mid_offset', dtype=tf.float32)

    val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
    val_loss_kpmaps = tf.keras.metrics.Mean('val_loss_kpmaps', dtype=tf.float32)
    val_loss_short_offset = tf.keras.metrics.Mean('val_loss_short_offset', dtype=tf.float32)
    val_loss_mid_offset = tf.keras.metrics.Mean('val_loss_mid_offset', dtype=tf.float32)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_log_dir = 'logs/' + current_time + '/val'
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    # determine start epoch in case the training has been stopped manually and resumed

    resume = last_step != 0 and (steps_per_epoch - last_step) != 0
    if resume:
        start_epoch = last_epoch
    else:
        start_epoch = last_epoch + 1

    # start processing

    for epoch in range(start_epoch, max_epochs + 1, 1):

        start = timer()

        print("Start processing epoch {}".format(epoch))

        # set the initial step index depending on if you resumed the processing

        if resume:
            step = last_step + 1
            data_iter = ds_train.skip(last_step)
            print(f"Skipping {last_step} steps (May take a few minutes)...")
            resume = False
        else:
            step = 0
            data_iter = ds_train

        # process steps
        accum_gradient = [tf.zeros_like(this_var) for this_var in model.trainable_variables]
        for x, y in data_iter:
            step += 1
            pred_data = PredictionData(tf.squeeze(y, axis = 0))
            y1 = tf.cast(tf.expand_dims(pred_data.kp_heatmaps(), axis=0), tf.float32)
            y2 = tf.cast(tf.expand_dims(pred_data.compute_short_offsets(), axis=0), tf.float32)
            y3 = tf.cast(tf.expand_dims(pred_data.compute_mid_offsets(), axis=0), tf.float32)
            grads, losses, total_loss = train_one_step(model, optimizer, x, [y1, y2, y3])
            accum_gradient = [(acum_grad+grad) for acum_grad, grad in zip(accum_gradient, grads)]

            train_loss(total_loss)
            train_loss_kpmaps(losses[0])
            train_loss_short_offset(losses[1])
            train_loss_mid_offset(losses[2])
            
            if step % 10 == 0:
                tf.print('Epoch', epoch, f'Step {step}/{steps_per_epoch}', 'kp_map_loss', losses[0], 'short_offset_loss', losses[1],
                         'mid_offset_loss', losses[2], 'Total loss', total_loss)
                
                with train_summary_writer.as_default():
                    summary_step = (epoch - 1) * steps_per_epoch + step - 1
                    tf.summary.scalar('loss', train_loss.result(), step=summary_step)
                    tf.summary.scalar('loss_kp_map', train_loss_kpmaps.result(), step=summary_step)
                    tf.summary.scalar('loss_short_offset', train_loss_short_offset.result(), step=summary_step)
                    tf.summary.scalar('train_loss_mid_offset', train_loss_mid_offset.result(), step=summary_step)

            if step % 50 ==0:
                accum_gradient = [this_grad/50 for this_grad in accum_gradient]
                optimizer.apply_gradients(zip(accum_gradient,model.trainable_variables))
                accum_gradient = [tf.zeros_like(this_var) for this_var in model.trainable_variables]  

            if step % 10000 == 0:
                figure = probe_model(model, test_img_path="man.jpg")
                with train_summary_writer.as_default():
                    tf.summary.image("Test prediction", plot_to_image(figure), step=step)

            if step % 1000 == 0:
                ckpt.step.assign(step)
                ckpt.epoch.assign(epoch)
                save_path = manager.save()
                print("Saved checkpoint for step {}: {}".format(step, save_path))

            if step >= steps_per_epoch:
                break

        print("Completed epoch {}. Saving weights...".format(epoch))
        model.save_weights(output_weights, overwrite=True)

        # save checkpoint at the end of an epoch
        ckpt.step.assign(step)
        ckpt.epoch.assign(epoch)
        manager.save()

        # reset metrics every epoch
        train_loss.reset_states()
        train_loss_kpmaps.reset_states()
        train_loss_short_offset.reset_states()
        train_loss_mid_offset.reset_states()
        
        end = timer()

        print("Epoch training time: " + str(timedelta(seconds=end - start)))

        # calculate validation loss
        print("Calculating validation losses...")
        for val_step, (x_val, y_val_true) in enumerate(ds_val):
            
            if val_step % 1000 == 0:
                print(f"Validation step {val_step} ...")
            val_pred_data = PredictionData(tf.squeeze(y_val_true, axis = 0))
            y_val_true1 = tf.cast(tf.expand_dims(val_pred_data.kp_heatmaps(), axis=0), tf.float32)
            y_val_true2 = tf.cast(tf.expand_dims(val_pred_data.compute_short_offsets(), axis=0), tf.float32)
            y_val_true3 = tf.cast(tf.expand_dims(val_pred_data.compute_mid_offsets(), axis=0), tf.float32)    
            
            val_true = [y_val_true1, y_val_true2, y_val_true3]
            
            y_val_pred = model(x_val)
            losses = get_losses(val_true, y_val_pred)
            
            total_loss = tf.reduce_sum(losses)
            val_loss(total_loss)
            val_loss_kpmaps(losses[0])
            val_loss_short_offset(losses[1])
            val_loss_mid_offset(losses[2])

        val_loss_res = val_loss.result()
        val_loss_kpmaps_res = val_loss_kpmaps.result()
        val_loss_short_offset_res = val_loss_short_offset.result()
        val_loss_mid_offset_res = val_loss_mid_offset.result()

        print(f'Validation losses for epoch: {epoch} : Loss short_offset {val_loss_short_offset_res}, mid_offset'
              f'{val_loss_mid_offset_res}, Loss kpmap {val_loss_kpmaps_res}, Total loss {val_loss_res}')

        with val_summary_writer.as_default():
            tf.summary.scalar('val_loss', val_loss_res, step=epoch)
            tf.summary.scalar('val_loss_kpmaps', val_loss_kpmaps_res, step=epoch)
            tf.summary.scalar('val_loss_short_offset', val_loss_short_offset_res, step=epoch)
            tf.summary.scalar('val_loss_mid_offset', val_loss_short_offset_res, step=epoch)
        val_loss.reset_states()
        val_loss_kpmaps.reset_states()
        val_loss_short_offset.reset_states()
        val_loss_mid_offset.reset_states()
        
model = model.model()
model.summary()
optimizer = Adam(lr)

# loading previous state if required
ckpt = tf.train.Checkpoint(step=tf.Variable(0), epoch=tf.Variable(0), optimizer=optimizer, net=model)
manager = tf.train.CheckpointManager(ckpt, checkpoints_folder, max_to_keep=3)
ckpt.restore(manager.latest_checkpoint)
last_step = int(ckpt.step)
last_epoch = int(ckpt.epoch)

if manager.latest_checkpoint:
    print(f"Restored from {manager.latest_checkpoint}")
    print(f"Resumed from epoch {last_epoch}, step {last_step}")
else:
    print("Initializing from scratch.")

    path = "base_model_weights/posenetBaseModel.h5" # get weights from person detection model
    model.load_weights(path, by_name=True,skip_mismatch=True)
    for l in range(len(model.layers)):
        if l <=82:
            model.layers[l].trainable = True
    
    
   


max_epochs = config.MAX_EPOCHS

#training loop
train(ds_train, ds_val, model, optimizer, ckpt, last_epoch, last_step,
          max_epochs, steps_per_epoch)
