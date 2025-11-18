import tensorflow as tf
import os

from val import run_validation

class Encoder(tf.keras.Model):
    
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim
        )
        self.lstm = tf.keras.layers.LSTM(
            units=hidden_size,
            return_state=True,
            return_sequences=False
        )

    def call(self, input_tokens):
        x = self.embedding(input_tokens)
        output, state_h, state_c = self.lstm(x)
        return output, state_h, state_c

class Decoder(tf.keras.Model):
    
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim
        )
        self.lstm = tf.keras.layers.LSTM(
            units=hidden_size,
            return_sequences=True,
            return_state=True
        )
        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, input_tokens, initial_states):
        x = self.embedding(input_tokens)
        outputs, state_h, state_c = self.lstm(x, initial_state=initial_states)

        # Project LSTM outputs to vocabulary logits
        logits = self.fc(outputs)

        return logits, state_h, state_c

class Seq2Seq(tf.keras.Model):
    
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        embedding_dim,
        hidden_size,
        src_tokenizer,
        tgt_tokenizer,
        bos_id,
        eos_id,
        max_decode_len=100
    ):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, embedding_dim, hidden_size)
        self.decoder = Decoder(tgt_vocab_size, embedding_dim, hidden_size)

        self.src_tok = src_tokenizer
        self.tgt_tok = tgt_tokenizer

        self.bos_id = bos_id
        self.eos_id = eos_id
        self.max_decode_len = max_decode_len

        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(0.001)

    def call(self, source_tokens, target_tokens):
        _, h, c = self.encoder(source_tokens)
        decoder_logits, _, _ = self.decoder(target_tokens, (h, c))
        return decoder_logits
    
    def _train_batch(self, src_batch, tgt_batch):

        # decoder_in = BOS + tokens[:-1]
        decoder_in = tgt_batch[:, :-1]

        # decoder_out = tokens[1:] + EOS
        decoder_out = tgt_batch[:, 1:]

        with tf.GradientTape() as tape:
            logits = self(src_batch, decoder_in)
            loss = self.loss_fn(decoder_out, logits)

        # Compute gradients and apply optimizer
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Compute accuracy
        preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(preds, decoder_out), tf.float32)
        )

        return loss, accuracy
    
    def train(
        self,
        token_pairs,
        batch_size=32,
        epochs=3,
        pad_id=0,
        val_path=None
    ):

        # Extract source and target sequences
        src_sequences = [p[0] for p in token_pairs]
        tgt_sequences = [p[1] for p in token_pairs]

        # Compute max lengths for padding
        max_src = max(len(s) for s in src_sequences)
        max_tgt = max(len(t) for t in tgt_sequences)

        # Pad sequences with pad_token
        src_padded = tf.keras.preprocessing.sequence.pad_sequences(
            src_sequences, maxlen=max_src, padding='post', value=pad_id
        )
        tgt_padded = tf.keras.preprocessing.sequence.pad_sequences(
            tgt_sequences, maxlen=max_tgt, padding='post', value=pad_id
        )

        # Create TensorFlow dataset from padded sequences
        dataset = tf.data.Dataset.from_tensor_slices((src_padded, tgt_padded))
        dataset = dataset.shuffle(20000).batch(batch_size)

        # Training loop
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")

            # Split epoch into batches
            for step, (src_batch, tgt_batch) in enumerate(dataset, start=1):

                # Perform a training step
                loss, acc = self._train_batch(src_batch, tgt_batch)

            # Calculate performance on validation set every 2 epochs
            if val_path and epoch % 1 == 0:
                print("\nRunning validation...")
                run_validation(val_path, self)

    
    def export(self, export_dir):
    
        # Create export directory if it doesn't exist
        os.makedirs(export_dir, exist_ok=True)

        # Save encoder and decoder weights separately
        encoder_path = os.path.join(export_dir, "encoder_weights")
        decoder_path = os.path.join(export_dir, "decoder_weights")

        self.encoder.save_weights(encoder_path)
        self.decoder.save_weights(decoder_path)

    def load(self, export_dir):

        # Load encoder and decoder weights separately
        encoder_path = os.path.join(export_dir, "encoder_weights")
        decoder_path = os.path.join(export_dir, "decoder_weights")

        self.encoder.load_weights(encoder_path)
        self.decoder.load_weights(decoder_path)

    def translate(self, text):
        
        # Tokenize input text
        src_ids = self.src_tok.encode(text)
        src_tensor = tf.constant([src_ids], dtype=tf.int32)

        # Run input text through encoder
        _, h, c = self.encoder(src_tensor)

        # Initialize decoder with BOS token
        next_token = tf.constant([[self.bos_id]], dtype=tf.int32)

        # Run decoder until EOS token or max_decode_len is reached
        output_tokens = []
        for _ in range(self.max_decode_len):

            # Run decoder
            logits, h, c = self.decoder(next_token, (h, c))

            # Extract predicted token
            next_id = tf.argmax(logits[0, 0]).numpy()
            
            # Check for EOS token
            if next_id == self.eos_id:
                break

            # Add token to output sequence
            output_tokens.append(int(next_id))

            # Feed token back into the decoder
            next_token = tf.constant([[next_id]], dtype=tf.int32)

        # Decode output tokens into text
        output_sentence = self.tgt_tok.decode(output_tokens)

        return output_sentence
