import argparse

import tensorflow as tf


class ToyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        tf.print(inputs)
        return self.dense(inputs)



def dataset_fn() -> tf.data.Dataset:
    features = tf.data.Dataset.from_tensor_slices(tf.expand_dims(tf.range(9, dtype=tf.float32), axis=-1))
    labels = tf.data.Dataset.from_tensor_slices(tf.expand_dims(tf.range(9, dtype=tf.int64), axis=-1))
    return tf.data.Dataset.zip((features, labels)).batch(3).repeat()


def main(distributed_type: str):
    assert distributed_type in ["sync", "ps"]
    
    if distributed_type == "sync":
        strat = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    else:
        resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
        if resolver.task_type in ("worker", "ps"):
            server = tf.distribute.Server(
                resolver.cluster_spec(),
                job_name=resolver.task_type,
                task_index=resolver.task_id,
                protocol=resolver.rpc_layer,
                start=True,
            )
            server.join()
        
        strat = tf.distribute.experimental.ParameterServerStrategy(resolver)
    
    with strat.scope():
        model = ToyModel()
        model.compile(loss="mse", optimizer="adam")

    model.fit(dataset_fn(), epochs=4, steps_per_epoch=20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--distributed-type", choices=["sync", "ps"])
    args = parser.parse_args()
    main(args.distributed_type)
