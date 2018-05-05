end_symbol = '$'
padding_symbol = '#'
start_symbol="^"
word2id = {symbol:i for i, symbol in enumerate('^$#abcdefghijklmnopqrstuvwxyz 0123456789+-')}
id2word = {i:symbol for symbol, i in word2id.items()}
max_len = 30
def load_model():
	sess = tf.Session()
	new_saver = tf.train.import_meta_graph('model/chatbot_model.meta')
	new_saver.restore(sess, tf.train.latest_checkpoint('model/'))
	sess.run(tf.local_variables_initializer())
	return sess
def reply(question, word2id, max_len, id2word, session):
    input_batch = tf.get_default_graph().get_tensor_by_name("input_batch:0")
    input_batch_len = tf.get_default_graph().get_tensor_by_name("input_batch_lengths:0")
    infer_predictions = tf.get_default_graph().get_tensor_by_name(
        "decode_1/decoder/transpose_1:0")

    question = text_prepare(question)
    ids, ids_len = sentence_to_ids(question, word2id, padded_len=max_len)
    ids = np.array(ids).reshape(1, len(ids))

    ids_len = np.array(ids_len).reshape(1)
    predictions = session.run([
        infer_predictions
    ], feed_dict={input_batch: ids, input_batch_len: ids_len})[0]
    return "".join(ids_to_sentence(predictions[0], id2word)).replace("$", "").capitalize()
