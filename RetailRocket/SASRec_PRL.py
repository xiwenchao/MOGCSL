import tensorflow as tf
import numpy as np
import pandas as pd
import os
import argparse
# import trfl
from utility import *
from SASRecModules import *
import logging
import time as Time

logging.getLogger().setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Run SASRec with PRL.")

    parser.add_argument('--epoch', type=int, default=30,
                        help='Number of max epochs.')
    parser.add_argument('--data', nargs='?', default='data',
                        help='data directory')
    # parser.add_argument('--pretrain', type=int, default=1,
    #                     help='flag for pretrain. 1: initialize from pretrain; 0: randomly initialize; -1: save the model to pretrain file')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--r_click', type=float, default=0.2,
                        help='reward for the click behavior.')
    parser.add_argument('--r_buy', type=float, default=1.0,
                        help='reward` for the purchase behavior.')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate.')
    parser.add_argument('--max_step', type=int, default=50,
                        help='maximum step')
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--num_blocks', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.3, type=float)
    parser.add_argument('--reward_to_go', default=2.0, type=float)
    parser.add_argument('--std', default=0.0, type=float)

    return parser.parse_args()


class SASRec_PRLnetwork:
    def __init__(self, hidden_size, learning_rate, item_num, state_size, max_step):
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.max_step = max_step

        all_embeddings = self.initialize_embeddings()

        self.inputs = tf.placeholder(tf.int32, [None, state_size], name='inputs')
        # self.cls=tf.placeholder(tf.int32, [None], name='cls')
        self.len_state = tf.placeholder(tf.int32, [None], name='len_state')
        self.target = tf.placeholder(tf.int32, [None], name='target')  # target item, to calculate ce loss
        self.reward_to_go = tf.placeholder(tf.float32, [None], name='reward_to_go')
        self.timestamp = tf.placeholder(tf.int32, [None], name='timestamp')
        self.is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
        self.weight=tf.placeholder(tf.float32, [None], name='weight')

        self.state_embeddings = tf.nn.embedding_lookup(all_embeddings['state_embeddings'], self.inputs)

        self.input_emb = tf.nn.embedding_lookup(all_embeddings['state_embeddings'], self.inputs)
        # Positional Encoding
        pos_emb = tf.nn.embedding_lookup(all_embeddings['pos_embeddings'],
                                         tf.tile(tf.expand_dims(tf.range(tf.shape(self.inputs)[1]), 0),
                                                 [tf.shape(self.inputs)[0], 1]))
        self.seq = self.input_emb + pos_emb

        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.inputs, item_num)), -1)
        # Dropout
        self.seq = tf.layers.dropout(self.seq,
                                     rate=args.dropout_rate,
                                     training=tf.convert_to_tensor(self.is_training))
        self.seq *= mask

        # Build blocks

        for i in range(args.num_blocks):
            with tf.variable_scope("num_blocks_%d" % i):
                # Self-attention
                self.seq = multihead_attention(queries=normalize(self.seq),
                                               keys=self.seq,
                                               num_units=self.hidden_size,
                                               num_heads=args.num_heads,
                                               dropout_rate=args.dropout_rate,
                                               is_training=self.is_training,
                                               causality=True,
                                               scope="self_attention_rec",
                                               # reuse=tf.AUTO_REUSE
                                               )

                # Feed forward
                self.seq = feedforward(normalize(self.seq), num_units=[self.hidden_size, self.hidden_size],
                                       dropout_rate=args.dropout_rate,
                                       is_training=self.is_training,
                                       scope='fc_rec')
                self.seq *= mask

        self.seq = normalize(self.seq)
        self.state_env = extract_axis_1(self.seq, self.len_state - 1)

        self.step_embeddings = tf.nn.embedding_lookup(all_embeddings['step_embeddings'], self.timestamp, name="timestamp_embedding_1")
        self.reward_embedding = tf.matmul(tf.expand_dims(self.reward_to_go, axis=-1),
                                          all_embeddings['reward_embedding'], name="reward_embedding_1")
        # self.cls_embedding=tf.nn.embedding_lookup(all_embeddings['cls_embedding'], self.cls)

        self.stack = tf.stack([self.state_env, self.step_embeddings, self.reward_embedding], axis=1)

        # Dropout
        self.stack = tf.layers.dropout(self.stack,
                                     rate=args.dropout_rate,
                                     training=tf.convert_to_tensor(self.is_training))

        # Build blocks

        for i in range(args.num_blocks):
            with tf.variable_scope("num_blocks_%d" % i):
                # Self-attention
                self.stack = multihead_attention(queries=self.stack,
                                               keys=self.stack,  ## Shijun: self.seq -> self.stack
                                               num_units=self.hidden_size,
                                               num_heads=args.num_heads,
                                               dropout_rate=args.dropout_rate,
                                               is_training=self.is_training,
                                               causality=False,
                                               scope="self_attention_rl",
                                               # reuse=tf.AUTO_REUSE
                                            )

                # Feed forward
                # self.seq = feedforward(normalize(self.seq), num_units=[self.hidden_size, self.hidden_size],
                #                        dropout_rate=args.dropout_rate,
                #                        is_training=self.is_training)

        # self.stack = normalize(self.stack)

        # s, t, r = tf.unstack(self.stack, axis=1)
        # self.state_final = s
        # self.state_final=tf.reduce_mean(self.seq,axis=1)

        s = tf.layers.dense(tf.transpose(self.stack, [0, 2, 1]), 1, activation=tf.nn.relu, name="dense_0")   # with(out) relu
        self.state_final = tf.squeeze(s, axis=-1, name="state_final")

        output = tf.contrib.layers.fully_connected(self.state_final, self.item_num, activation_fn=None,
                                                        scope='fc')

        self.output = tf.add(output, 0, name="output")
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target, logits=self.output)
        self.loss = tf.reduce_mean(self.loss)
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def initialize_embeddings(self):
        all_embeddings = dict()
        state_embeddings = tf.Variable(tf.random_normal([self.item_num + 1, self.hidden_size], 0.0, 0.01),
                                       name='state_embeddings')
        step_embeddings = tf.Variable(tf.random_normal([self.max_step, self.hidden_size], 0.0, 0.01),
                                      name='step_embeddings')
        reward_embedding = tf.Variable(tf.random_normal([1, self.hidden_size], 0.0, 0.01),
                                       name='reward_embedding')
        pos_embeddings = tf.Variable(tf.random_normal([self.state_size, self.hidden_size], 0.0, 0.01),
                                     name='pos_embeddings')
        all_embeddings['pos_embeddings'] = pos_embeddings
        # cls_embedding = tf.Variable(tf.random_normal([1, self.hidden_size], 0.0, 0.01),
        #                                name='cls_embedding')
        all_embeddings['state_embeddings'] = state_embeddings
        all_embeddings['step_embeddings'] = step_embeddings
        all_embeddings['reward_embedding'] = reward_embedding
        # all_embeddings['cls_embedding']=cls_embedding
        return all_embeddings


def evaluate(sess, epoch_num):
    eval_sessions = pd.read_pickle(os.path.join(data_directory, 'sampled_val.df'))

    eval_sessions['valid_session'] = eval_sessions.session_id.map(
        eval_sessions.groupby('session_id')['item_id'].size() < max_step + 1)
    eval_sessions = eval_sessions.loc[eval_sessions.valid_session].drop('valid_session', axis=1)

    eval_ids = eval_sessions.session_id.unique()
    groups = eval_sessions.groupby('session_id')
    batch = 100
    evaluated = 0
    total_clicks = 0.0
    total_purchase = 0.0
    total_reward = [0, 0, 0, 0]
    hit_clicks = [0, 0, 0, 0]
    ndcg_clicks = [0, 0, 0, 0]
    hit_purchase = [0, 0, 0, 0]
    ndcg_purchase = [0, 0, 0, 0]
    while evaluated < len(eval_ids):
        states, len_states, actions, rewards = [], [], [], []
        timestamp, reward_to_go = [], []
        # cls=[]
        for i in range(batch):
            # if evaluated in eval_ids:
            id = eval_ids[evaluated]
            group = groups.get_group(id)
            history = []
            step = 0
            for index, row in group.iterrows():
                state = list(history)
                len_states.append(state_size if len(state) >= state_size else 1 if len(state) == 0 else len(state))
                state = pad_history(state, state_size, item_num)
                states.append(state)
                action = row['item_id']
                is_buy = row['is_buy']
                reward = reward_buy if is_buy == 1 else reward_click
                if is_buy == 1:
                    total_purchase += 1.0
                else:
                    total_clicks += 1.0
                actions.append(action)
                rewards.append(reward)
                timestamp.append(step)
                # r_t = r_t_g
                r_t = np.random.normal(r_t_g, args.std) * reward_mean[step] 
                reward_to_go.append(r_t)
                history.append(row['item_id'])
                # cls.append(0)
                step += 1
            evaluated += 1
            if evaluated == len(eval_ids):
                break
        prediction = sess.run(SASRec_PRLnet.output, feed_dict={SASRec_PRLnet.inputs: states, SASRec_PRLnet.len_state: len_states,
                                                            SASRec_PRLnet.reward_to_go: reward_to_go,
                                                            SASRec_PRLnet.timestamp: timestamp,
                                                            # SASRec_PRLnet.cls: cls,
                                                            SASRec_PRLnet.is_training: False
                                                            })
        sorted_list = np.argsort(prediction)
        calculate_hit(sorted_list, topk, actions, rewards, reward_click, total_reward, hit_clicks, ndcg_clicks,
                      hit_purchase, ndcg_purchase)
    logging.info('#############################################################')
    logging.info('Start evaluate at {} epoch!'.format(epoch_num))
    logging.info('total clicks: %d, total purchase:%d' % (total_clicks, total_purchase))
    for i in range(len(topk)):
        hr_click = hit_clicks[i] / total_clicks
        hr_purchase = hit_purchase[i] / total_purchase
        ng_click = ndcg_clicks[i] / total_clicks
        ng_purchase = ndcg_purchase[i] / total_purchase
        logging.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        logging.info('cumulative reward @ %d: %f' % (topk[i], total_reward[i]))
        logging.info('clicks hr ndcg @ %d : %f, %f' % (topk[i], hr_click, ng_click))
        logging.info('purchase hr and ndcg @%d : %f, %f' % (topk[i], hr_purchase, ng_purchase))
    logging.info('#############################################################')


# Shijun:
def test(sess, epoch_num):
    test_sessions = pd.read_pickle(os.path.join(data_directory, 'sampled_test.df'))

    test_sessions['valid_session'] = test_sessions.session_id.map(
        test_sessions.groupby('session_id')['item_id'].size() < max_step + 1)
    test_sessions = test_sessions.loc[test_sessions.valid_session].drop('valid_session', axis=1)

    test_ids = test_sessions.session_id.unique()
    groups = test_sessions.groupby('session_id')
    batch = 100
    tested = 0
    total_clicks = 0.0
    total_purchase = 0.0
    total_reward = [0, 0, 0, 0]
    hit_clicks = [0, 0, 0, 0]
    ndcg_clicks = [0, 0, 0, 0]
    hit_purchase = [0, 0, 0, 0]
    ndcg_purchase = [0, 0, 0, 0]
    while tested < len(test_ids):
        states, len_states, actions, rewards = [], [], [], []
        timestamp, reward_to_go = [], []
        # cls=[]
        for i in range(batch):
            # if evaluated in eval_ids:
            id = test_ids[tested]
            group = groups.get_group(id)
            history = []
            step = 0
            for index, row in group.iterrows():
                state = list(history)
                len_states.append(state_size if len(state) >= state_size else 1 if len(state) == 0 else len(state))
                state = pad_history(state, state_size, item_num)
                states.append(state)
                action = row['item_id']
                is_buy = row['is_buy']
                reward = reward_buy if is_buy == 1 else reward_click
                if is_buy == 1:
                    total_purchase += 1.0
                else:
                    total_clicks += 1.0
                actions.append(action)
                rewards.append(reward)
                timestamp.append(step)
                # r_t = r_t_g
                r_t = np.random.normal(r_t_g, args.std) * reward_mean[step] 
                reward_to_go.append(r_t)
                # reward_to_go.append(np.random.normal(r_t_g, args.std))
                history.append(row['item_id'])
                # cls.append(0)
                step += 1
            tested += 1
            if tested == len(test_ids):
                break
        prediction = sess.run(SASRec_PRLnet.output, feed_dict={SASRec_PRLnet.inputs: states, SASRec_PRLnet.len_state: len_states,
                                                            SASRec_PRLnet.reward_to_go: reward_to_go,
                                                            SASRec_PRLnet.timestamp: timestamp,
                                                            # SASRec_PRLnet.cls: cls,
                                                            SASRec_PRLnet.is_training: False
                                                            })
        sorted_list = np.argsort(prediction)
        calculate_hit(sorted_list, topk, actions, rewards, reward_click, total_reward, hit_clicks, ndcg_clicks,
                      hit_purchase, ndcg_purchase)
    logging.info('#############################################################')
    logging.info('Start test at {} epoch!'.format(epoch_num))
    logging.info('total clicks: %d, total purchase:%d' % (total_clicks, total_purchase))
    for i in range(len(topk)):
        hr_click = hit_clicks[i] / total_clicks
        hr_purchase = hit_purchase[i] / total_purchase
        ng_click = ndcg_clicks[i] / total_clicks
        ng_purchase = ndcg_purchase[i] / total_purchase
        logging.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        logging.info('cumulative reward @ %d: %f' % (topk[i], total_reward[i]))
        logging.info('clicks hr ndcg @ %d : %f, %f' % (topk[i], hr_click, ng_click))
        logging.info('purchase hr and ndcg @%d : %f, %f' % (topk[i], hr_purchase, ng_purchase))
    logging.info('#############################################################')


if __name__ == '__main__':
    # Network parameters
    args = parse_args()

    data_directory = args.data
    data_statis = pd.read_pickle(
        os.path.join(data_directory, 'data_statis_PRL.df'))  # read data statistics, includeing state_size and item_num, mean_reward and max_reward
    state_size = data_statis['state_size'][0]  # the length of history to define the state
    item_num = data_statis['item_num'][0]  # total number of items
    reward_mean = data_statis['reward_mean'][0]
    reward_max = data_statis['reward_max'][0]
    reward_click = args.r_click
    reward_buy = args.r_buy
    topk = [1, 5, 10, 20]
    max_step = args.max_step
    r_t_g = args.reward_to_go
    # save_file = 'pretrain-SASRec/%d' % (hidden_size)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction=0.45
    config.gpu_options.allow_growth = True

    tf.reset_default_graph()

    logging.basicConfig(
        filename="./log/SASRec_PRL/head:{}_block:{}_drop:{}_rtg:{}_std:{}_lr:{}_{}".format(
                                                            args.num_heads, args.num_blocks, args.dropout_rate,
                                                            args.reward_to_go, args.std,args.lr, 
                                                            Time.strftime("%m-%d %H:%M:%S", Time.localtime())))

    SASRec_PRLnet = SASRec_PRLnetwork(hidden_size=args.hidden_factor, learning_rate=args.lr, item_num=item_num,
                                state_size=state_size, max_step=max_step)

    trajectory_buffer = pd.read_pickle(os.path.join(data_directory, 'trajectory_buffer_PRL.df'))
    saver = tf.train.Saver(max_to_keep = 30)

    total_step = 0
    with tf.Session(config=config) as sess:
        # Initialize variables
        sess.run(tf.global_variables_initializer())
        # evaluate(sess)
        num_rows = trajectory_buffer.shape[0]
        num_batches = int(num_rows / args.batch_size)
        for i in range(args.epoch):
            current_time = Time.time()
            for j in range(num_batches):
                batch = trajectory_buffer.sample(n=args.batch_size).to_dict()
                state = list(batch['state'].values())
                len_state = list(batch['len_state'].values())
                target = list(batch['action'].values())
                timestamp = list(batch['timestamp'].values())
                reward_to_go = list(batch['reward_to_go'].values())
                is_buy=list(batch['is_buy'].values())
                weight=[]
                for k in range(args.batch_size):
                    # reward_to_go[k] = reward_to_go[k] / reward_mean[timestamp[k]]
                    if is_buy[k]==1:
                        weight.append(reward_buy)
                    if is_buy[k]==0:
                        weight.append(reward_click)
                # cls=[0]*len(timestamp)
                loss, _ = sess.run([SASRec_PRLnet.loss, SASRec_PRLnet.opt],
                                   feed_dict={SASRec_PRLnet.inputs: state,
                                              SASRec_PRLnet.len_state: len_state,
                                              SASRec_PRLnet.target: target,
                                              SASRec_PRLnet.reward_to_go: reward_to_go,
                                              SASRec_PRLnet.timestamp: timestamp,
                                              SASRec_PRLnet.weight:weight,
                                              # SASRec_PRLnet.cls:cls,
                                              SASRec_PRLnet.is_training: True})
                total_step += 1
                if total_step % 200 == 0:
                    # print("the loss in %dth batch is: %f" % (total_step, loss))
                    logging.info("the loss in %dth batch is: %f" % (total_step, loss))           
            end_time = Time.time()
            logging.info("the time in %dth epoch is: %f" % (i, end_time - current_time))
            logging.info("the number of parameters is: %d" % np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
            evaluate(sess, i)
            save_file = "./model/SASRec_PRL/epoch_{}_{}".format(i, Time.strftime("%m-%d-%H:%M:%S", Time.localtime()))
            saver.save(sess, save_file)
