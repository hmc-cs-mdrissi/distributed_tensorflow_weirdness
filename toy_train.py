import tensorflow as tf
from tensorflow.keras.utils.experimental import DatasetCreator
from tensorflow.python.distribute.coordinator import coordinator_context


class ToyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        tf.print(inputs)
        return self.dense(inputs)


def dataset_fn(_: tf.distribute.InputContext) -> tf.data.Dataset:
    # def call_time_worker_index() -> int:
    #     dispatch_context = coordinator_context.get_current_dispatch_context()
    #     return dispatch_context.worker_index

    # graph = tf.compat.v1.get_default_graph()
    # worker_index = graph.capture_call_time_value(call_time_worker_index, tf.TensorSpec([], dtype=tf.dtypes.int64))
    # worker_index.op._set_attr("_user_specified_name", tf.compat.v1.AttrValue(s=tf.compat.as_bytes("worker_index")))

    # num_workers = tf.distribute.get_strategy()._extended._num_workers + 1

    # print("Num workers:", num_workers)
    # print("Worker index:", worker_index)
    # tf.print("Number of workers: ", num_workers)
    # tf.print("Worker index: ", worker_index)

    features = tf.data.Dataset.from_tensor_slices(tf.expand_dims(tf.range(9, dtype=tf.float32), axis=-1))
    labels = tf.data.Dataset.from_tensor_slices(tf.expand_dims(tf.range(9, dtype=tf.int64), axis=-1))
    dataset = tf.data.Dataset.zip((features, labels)).batch(3)

    # dataset = dataset.shard(num_workers, worker_index)
    return dataset.repeat()


def main():
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

    model.fit(DatasetCreator(dataset_fn), epochs=4, steps_per_epoch=20)


if __name__ == "__main__":
    main()
