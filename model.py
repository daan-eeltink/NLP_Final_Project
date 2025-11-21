import tensorflow as tf
import os



class Encoder(tf.keras.Model):
    
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()

        # Define embedding and LSTM layers
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

        # Embed input tokens
        x = self.embedding(input_tokens)

        # Process embeddings through LSTM
        output, state_h, state_c = self.lstm(x)

        return output, state_h, state_c



class Decoder(tf.keras.Model):
    
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()

        # Define embedding, LSTM, and Dense layers
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

        # Embed input tokens
        x = self.embedding(input_tokens)

        # Process embeddings through LSTM with given initial states
        outputs, state_h, state_c = self.lstm(x, initial_state=initial_states)

        # Convert LSTM outputs to vocabulary logits
        logits = self.fc(outputs)

        return logits, state_h, state_c



class Seq2Seq(tf.keras.Model):
    
    def __init__(
        self,
        src_tokenizer,
        tgt_tokenizer,
        embedding_dim = 256,
        hidden_size = 512,
        max_decode_len=100
    ):
        super().__init__()

        # Initialize encoder and decoder
        self.encoder = Encoder(src_tokenizer.get_piece_size(), embedding_dim, hidden_size)
        self.decoder = Decoder(tgt_tokenizer.get_piece_size(), embedding_dim, hidden_size)

        # Store tokenizers and special token IDs
        self.src_tok = src_tokenizer
        self.tgt_tok = tgt_tokenizer

        self.bos_id = tgt_tokenizer.bos_id()
        self.eos_id = tgt_tokenizer.eos_id()
        self.max_decode_len = max_decode_len

        # Define loss function and optimizer
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(0.001)

    def call(self, source_tokens, target_tokens):

        # Encode source tokens
        _, h, c = self.encoder(source_tokens)

        # Decode teacher forced target tokens
        decoder_logits, _, _ = self.decoder(target_tokens, (h, c))

        return decoder_logits
    
    def _train_batch(self, src_batch, tgt_batch):

        # decoder_in = BOS + tokens[:-1]
        decoder_in = tgt_batch[:, :-1]

        # decoder_out = tokens[1:] + EOS
        decoder_out = tgt_batch[:, 1:]

        # Keep track of gradients
        with tf.GradientTape() as tape:
            logits = self(src_batch, decoder_in)
            loss = self.loss_fn(decoder_out, logits)

        # Compute gradients and apply optimizer
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return loss
    
    def train(
        self,
        token_pairs,
        epochs,
        batch_size=32,
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
            src_sequences, maxlen=max_src, padding='post', value=self.src_tok.pad_id()
        )
        tgt_padded = tf.keras.preprocessing.sequence.pad_sequences(
            tgt_sequences, maxlen=max_tgt, padding='post', value=self.tgt_tok.pad_id()
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
                loss = self._train_batch(src_batch, tgt_batch)

                # Print loss every 10 steps
                if step % 10 == 0:
                    print(f" Step {step}, Loss: {loss.numpy():.4f}", end="\r")

    def export(self, export_dir, prefix="seq2seq_model"):
    
        # Create export directory if it doesn't exist
        os.makedirs(export_dir, exist_ok=True)

        # Save encoder and decoder weights separately
        encoder_path = os.path.join(export_dir, f"{prefix}_encoder.weights.h5")
        decoder_path = os.path.join(export_dir, f"{prefix}_decoder.weights.h5")

        self.encoder.save_weights(encoder_path)
        self.decoder.save_weights(decoder_path)

    def load(self, export_dir, prefix="seq2seq_model"):

        # Load encoder and decoder weights separately
        encoder_path = os.path.join(export_dir, f"{prefix}_encoder.weights.h5")
        decoder_path = os.path.join(export_dir, f"{prefix}_decoder.weights.h5")

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
