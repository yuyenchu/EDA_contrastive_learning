import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import ops
# from keras import saving

from utils import build_encoder, build_projection_head, build_classification_head

# @saving.register_keras_serializable()
class ContrastiveModel(keras.Model):
    def __init__(self, temperature=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.temperature = temperature
        self.encoder = build_encoder()
        # Non-linear MLP as projection head
        self.projection_head = build_projection_head()
        # Single dense layer for linear probing
        self.linear_probe = build_classification_head()

        # self.encoder.summary()
        # self.projection_head.summary()
        # self.linear_probe.summary()
        self.train_mode = 'contrastive'

    def call(self, inputs, training=False):
        features = self.encoder(inputs, training=training)
        out = self.linear_probe(features, training=training)
        return out
    
    def compile(self, optimizer, train_mode, **kwargs):
        super().compile(**kwargs)

        assert train_mode in ('contrastive', 'prediction')
        self.train_mode = train_mode
        self.optimizer = optimizer

        # self.contrastive_loss will be defined as a method
        self.probe_loss = keras.losses.BinaryCrossentropy()

        self.contrastive_loss_tracker = keras.metrics.Mean(name="c_loss") # infofce loss
        self.contrastive_sim = keras.metrics.Mean(name="c_sim") # average similarty score, should apporoach 1
        self.contrastive_dsim = keras.metrics.Mean(name="c_dsim") # average dissimilarty score, should apporoach 0
        self.probe_loss_tracker = keras.metrics.Mean(name="p_loss")
        self.probe_accuracy = keras.metrics.BinaryAccuracy(name="p_acc")

    @property
    def metrics(self):
        return [
            self.contrastive_loss_tracker,
            self.contrastive_sim,
            self.contrastive_dsim,
            self.probe_loss_tracker,
            self.probe_accuracy,
        ]

    def contrastive_loss(self, projections_1, projections_2):
        # InfoNCE loss (information noise-contrastive estimation)
        # Cosine similarity: the dot product of the l2-normalized feature vectors
        b = tf.shape(projections_1)[0]
        p1 = tf.math.l2_normalize(projections_1, axis=-1)
        p2 = tf.math.l2_normalize(projections_2, axis=-1)
        similarities = tf.linalg.matmul(p1, p2, transpose_b=True)
        diag = tf.linalg.tensor_diag_part(similarities)
        self.contrastive_sim.update_state(tf.reduce_mean(diag))
        self.contrastive_dsim.update_state((tf.reduce_sum(similarities,axis=0)-diag)/tf.cast(b-1, tf.float32))
        self.contrastive_dsim.update_state((tf.reduce_sum(similarities,axis=1)-diag)/tf.cast(b-1, tf.float32))

        similarities = (tf.math.exp(similarities) / self.temperature)
        diag = tf.linalg.tensor_diag_part(similarities)
        # symmetrized temperature-scaled similarities are used
        loss_1_2 = -tf.reduce_mean(tf.math.log(diag/(tf.reduce_sum(similarities,axis=0)-diag)))
        loss_2_1 = -tf.reduce_mean(tf.math.log(diag/(tf.reduce_sum(similarities,axis=1)-diag)))
        # The similarity between the representations of two augmented views of the
        # same eda segment should be higher than their similarity with other views
        return (loss_1_2 + loss_2_1) / 2

    def train_step(self, data):
        if (self.train_mode == 'contrastive'):
            eda1, eda2 = data
            # paired unlabeled edas are used
            with tf.GradientTape() as tape:
                features_1 = self.encoder(eda1, training=True)
                features_2 = self.encoder(eda2, training=True)
                # The representations are passed through a projection mlp
                projections_1 = self.projection_head(features_1, training=True)
                projections_2 = self.projection_head(features_2, training=True)
                contrastive_loss = self.contrastive_loss(projections_1, projections_2)
            gradients = tape.gradient(
                contrastive_loss,
                self.encoder.trainable_weights + self.projection_head.trainable_weights,
            )
            self.optimizer.apply_gradients(
                zip(
                    gradients,
                    self.encoder.trainable_weights + self.projection_head.trainable_weights,
                )
            )
            self.contrastive_loss_tracker.update_state(contrastive_loss)
            return {m.name: m.result() for m in self.metrics[:3]}
        elif (self.train_mode == 'prediction'):
            # Labels are only used in evalutation for an on-the-fly logistic regression
            eda, labels = data
            with tf.GradientTape() as tape:
                # the encoder is used in inference mode here to avoid regularization
                # and updating the batch normalization paramers if they are used
                features = self.encoder(eda, training=False)
                class_logits = self.linear_probe(features, training=True)
                probe_loss = self.probe_loss(labels, class_logits)
            gradients = tape.gradient(probe_loss, self.linear_probe.trainable_weights)
            self.optimizer.apply_gradients(
                zip(gradients, self.linear_probe.trainable_weights)
            )
            self.probe_loss_tracker.update_state(probe_loss)
            self.probe_accuracy.update_state(labels, class_logits)
            return {m.name: m.result() for m in self.metrics[-2:]}
        else:
            raise ValueError('train_mode not recognized:', self.train_mode)

    def test_step(self, data):
        eda, labels = data

        # For testing the components are used with a training=False flag
        features = self.encoder(eda, training=False)
        class_logits = self.linear_probe(features, training=False)
        probe_loss = self.probe_loss(labels, class_logits)
        self.probe_loss_tracker.update_state(probe_loss)
        self.probe_accuracy.update_state(labels, class_logits)

        # Only the probe metrics are logged at test time
        return {m.name: m.result() for m in self.metrics[-2:]}
