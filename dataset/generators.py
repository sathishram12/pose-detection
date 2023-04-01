import tensorflow as tf

from dataset.dataflows import get_dataflow

def get_dataset(annot_path, img_dir, batch_size, strict=False, x_size=336):
    def gen(df):
        def f():
            for i in df:
                yield tuple(i)
        return f

    df, size = get_dataflow(
        annot_path=annot_path,
        img_dir=img_dir,
        strict=strict,
        x_size=x_size
    )
    df.reset_state()

    ds = tf.data.Dataset.from_generator(
        gen(df), (tf.uint8, tf.float32),
        
    )

    ds = ds.map(lambda x0, x1: (x0, (x1)))
    ds = ds.batch(batch_size)

    return ds, size
