import tensorflow as tf
from read_utils import TextConverter
from model import CharRNN
import os
from IPython import embed

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('lstm_size', 128, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
tf.flags.DEFINE_boolean('use_embedding', True, 'whether to use embedding')
tf.flags.DEFINE_integer('embedding_size', 128, 'size of embedding')
tf.flags.DEFINE_string('converter_path', 'model/poetry/converter.pkl', 'model/name/converter.pkl')
tf.flags.DEFINE_string('checkpoint_path', 'model/poetry/', 'checkpoint path')
tf.flags.DEFINE_string('start_string', '前不见古人，后不见来者', 'use this string to start generating')
tf.flags.DEFINE_integer('max_length', 26, 'max length to generate')


def main(_):
#    FLAGS.start_string = FLAGS.start_string.decode('utf-8')  # python3.x下注释掉
    converter = TextConverter(filename=FLAGS.converter_path)
    if os.path.isdir(FLAGS.checkpoint_path):
        FLAGS.checkpoint_path =\
            tf.train.latest_checkpoint(FLAGS.checkpoint_path)

    model = CharRNN(converter.vocab_size, sampling=True,
                    lstm_size=FLAGS.lstm_size, num_layers=FLAGS.num_layers,
                    use_embedding=FLAGS.use_embedding,
                    embedding_size=FLAGS.embedding_size)

    model.load(FLAGS.checkpoint_path)  # 加载模型

    start = converter.text_to_arr(FLAGS.start_string)  # 开始生成
    arr = model.sample(FLAGS.max_length, start, converter.vocab_size)
    print(converter.arr_to_text(arr))  # 数组转换为文本


if __name__ == '__main__':
    tf.app.run()
