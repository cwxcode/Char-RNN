import tensorflow as tf
from read_utils import TextConverter, batch_generator
from model import CharRNN
import os
import codecs

FLAGS = tf.flags.FLAGS  # 在用命令行执行程序时，需要传些参数

tf.flags.DEFINE_string('name', 'poetry', 'name of the model')
tf.flags.DEFINE_integer('num_seqs', 32, 'number of seqs in one batch')
tf.flags.DEFINE_integer('num_steps', 26, 'length of one seq')
tf.flags.DEFINE_integer('lstm_size', 128, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
tf.flags.DEFINE_boolean('use_embedding', True, 'whether to use embedding')
tf.flags.DEFINE_integer('embedding_size', 128, 'size of embedding')
tf.flags.DEFINE_float('learning_rate', 0.005, 'learning_rate')
tf.flags.DEFINE_float('train_keep_prob', 0.5, 'dropout rate during training')
tf.flags.DEFINE_string('input_file', './data/poetry.txt', 'utf8 encoded text file')
tf.flags.DEFINE_integer('max_steps', 10000, 'max steps to train')
tf.flags.DEFINE_integer('save_every_n', 1000, 'save the model every n steps')
tf.flags.DEFINE_integer('log_every_n', 10, 'log to the screen every n steps')
tf.flags.DEFINE_integer('max_vocab', 3500, 'max char number')


def main(_):
    model_path = os.path.join('model', FLAGS.name)  # 保存模型的路径
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    # 用codecs提供的open方法来指定打开的文件的语言编码，它会在读取的时候自动转换为内部unicode
    with codecs.open(FLAGS.input_file, encoding='utf-8') as f:
        text = f.read()  # 读取训练的文本
    converter = TextConverter(text, FLAGS.max_vocab)  # 转换text文本格式
    converter.save_to_file(os.path.join(model_path, 'converter.pkl'))

    arr = converter.text_to_arr(text)  # 转换text为数组
    g = batch_generator(arr, FLAGS.num_seqs, FLAGS.num_steps)  # 批生成
    print(converter.vocab_size)
    model = CharRNN(converter.vocab_size,  # 读取模型
                    num_seqs=FLAGS.num_seqs,
                    num_steps=FLAGS.num_steps,
                    lstm_size=FLAGS.lstm_size,
                    num_layers=FLAGS.num_layers,
                    learning_rate=FLAGS.learning_rate,
                    train_keep_prob=FLAGS.train_keep_prob,
                    use_embedding=FLAGS.use_embedding,
                    embedding_size=FLAGS.embedding_size
                    )
    model.train(g,  # 训练
                FLAGS.max_steps,
                model_path,
                FLAGS.save_every_n,
                FLAGS.log_every_n,
                )


if __name__ == '__main__':
    tf.app.run()
