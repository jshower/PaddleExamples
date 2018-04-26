import os
import math
import time
import numpy as np
import paddle
import paddle.fluid as fluid

import reader
from network_conf_paral_ex import ner_net

def to_lodtensor(data, place):
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for l in seq_lens:
        cur_len += l
        lod.append(cur_len)
    flattened_data = np.concatenate(data, axis=0).astype("int64")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    res = fluid.LoDTensor()
    res.set(flattened_data, place)
    res.set_lod([lod])
    return res

def test2(chunk_evaluator, inference_program, test_data, place, cur_fetch_list):
    chunk_evaluator.reset()
    exe = fluid.Executor(place)
    for data in test_data():
        word = to_lodtensor(map(lambda x: x[0], data), place)
        mention = to_lodtensor(map(lambda x: x[1], data), place)
        target = to_lodtensor(map(lambda x: x[2], data), place)
        result_list = exe.run(inference_program,feed={"word": word,"mention": mention,"target": target},fetch_list=cur_fetch_list)
        number_infer = np.array(result_list[0])
        number_label = np.array(result_list[1])
        number_correct = np.array(result_list[2])
        #print "test2"
        #print type(number_infer[0])
        #print type(number_label[0])
        #print type(number_correct[0])
        #print number_infer
        chunk_evaluator.update(number_infer[0], number_label[0], number_correct[0])
    return chunk_evaluator.eval()
        #result_list = exe.run(fetch_list=cur_fetch_list,
        #              feed={"word": word,
        #                    "mention": mention,
        #                    "target": target}, fetch_list=cur_fetch_list)

def test(exe, chunk_evaluator, inference_program, test_data, place, cur_fetch_list):
    test_exe = fluid.ParallelExecutor(
        use_cuda=True,
        main_program=inference_program,
        share_vars_from=exe)
    chunk_evaluator.reset()
    for data in test_data():
        word = to_lodtensor(map(lambda x: x[0], data), place)
        mention = to_lodtensor(map(lambda x: x[1], data), place)
        target = to_lodtensor(map(lambda x: x[2], data), place)
        result_list = test_exe.run(fetch_list=cur_fetch_list,
                      feed={"word": word,
                            "mention": mention,
                            "target": target})
        number_infer = np.array(result_list[0])
        number_label = np.array(result_list[1])
        number_correct = np.array(result_list[2])
        print number_infer
        print number_label
        print number_correct
    	chunk_evaluator.update(number_infer.sum(), number_label.sum(), number_correct.sum())
    return chunk_evaluator.eval()

def main(train_data_file, test_data_file, vocab_file, target_file,
         model_save_dir, num_passes, use_gpu, parallel):
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    BATCH_SIZE = 256
    word_dict = reader.load_dict(vocab_file)
    word_dict_len = len(word_dict)
    label_dict = reader.load_dict(target_file)
    label_dict_len = len(label_dict)

    #print word_dict_len
    main = fluid.Program()
    startup = fluid.Program()
    with fluid.program_guard(main, startup):
        avg_cost, feature_out, word, mention, target = ner_net(word_dict_len, label_dict_len, parallel)

        sgd_optimizer = fluid.optimizer.SGD(learning_rate=1e-3)
        sgd_optimizer.minimize(avg_cost)

        crf_decode = fluid.layers.crf_decoding(
            input=feature_out, 
            param_attr=fluid.ParamAttr(name='crfw',
            #regularizer=fluid.regularizer.L2DecayRegularizer(regularization_coeff=1e-4)
        ))

        (precision, recall, f1_score, num_infer_chunks, num_label_chunks, num_correct_chunks) = fluid.layers.chunk_eval(
            input=crf_decode,
            label=target,
            chunk_scheme="IOB",
            num_chunk_types=int(math.ceil((label_dict_len - 1) / 2.0)))

        chunk_evaluator = fluid.metrics.ChunkEvaluator()

    	inference_program = fluid.default_main_program().clone()
    	with fluid.program_guard(inference_program):
            inference_program = fluid.io.get_inference_program([num_infer_chunks, num_label_chunks, num_correct_chunks])

        train_reader = paddle.batch(
            paddle.reader.shuffle(
                reader.file_reader(train_data_file), buf_size=2000000),
            batch_size=BATCH_SIZE)
        test_reader = paddle.batch(
            paddle.reader.shuffle(
                reader.file_reader(test_data_file), buf_size=2000000),
            batch_size=BATCH_SIZE)

        place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
        #place = fluid.CPUPlace()
        feeder = fluid.DataFeeder(feed_list=[word, mention, target], place=place)

        #train_reader_iter = train_reader()
        #data = train_reader_iter.next()
        #feed_dict = feeder.feed(data)
        exe = fluid.Executor(place)

        exe.run(startup)
        exe = fluid.ParallelExecutor(loss_name=avg_cost.name, use_cuda=True)
        batch_id = 0
        #print fluid.default_main_program()
        for pass_id in xrange(num_passes):
        
            chunk_evaluator.reset()
            train_reader_iter = train_reader()
            start_time = time.time()
            while True:
                try:
                    cur_batch = next(train_reader_iter)
                    cost, nums_infer, nums_label, nums_correct = exe.run(
                        fetch_list=[avg_cost.name, num_infer_chunks.name, num_label_chunks.name, num_correct_chunks.name],
                        #feed_dict=feed_dict
                        feed=feeder.feed(cur_batch))
                    chunk_evaluator.update(np.array(nums_infer).sum(), np.array(nums_label).sum(), np.array(nums_correct).sum())
                    cost_list = np.array(cost)
                    #cost_list = map(np.array,cost)[0]
                    #total_cost = 0
                    #for item in cost_list:
                    #    total_cost += item
                    #print("total_cost" + str(total_cost) + "cost:" + str(map(np.array,cost)[0]))
                    #save_dirname = os.path.join(model_save_dir, "params_pass_%d" % pass_id)
                    #fluid.io.save_inference_model(save_dirname, ['word', 'mention', 'target'],
                    #                      [crf_decode], exe)
                except StopIteration:
                    break
            end_time = time.time()
            print("pass_id" + str(pass_id) + "time_cost:" + str(end_time-start_time))
            precision, recall, f1_score = chunk_evaluator.eval()
            #print precision
            #print recall
            #print f1_score
            print("Pass" + str(pass_id) + "finished.!")
            #print("Pass" + str(pass_id) + "")
            pass_precision, pass_recall, pass_f1_score = test(
                exe, chunk_evaluator, inference_program, test_reader, place, [num_infer_chunks.name, num_label_chunks.name, num_correct_chunks.name])
            print("precision:" + str(pass_precision) + ", recall:" + str(pass_recall) + ", f1:" + str(pass_f1_score))
            p,r,f1 = test2(chunk_evaluator, inference_program, test_reader, place, [num_infer_chunks, num_label_chunks, num_correct_chunks])
            #print type(r)
            #print type(p)
            #print type(f1)
            print("precision:" + str(p) + ", recall:" + str(r) + ", f1:" + str(f1))
            #if batch_id % 5 == 0:
            #    print("Pass " + str(pass_id) + ", Batch " + str(
            #        batch_id) + ", Cost " + str(cost[0]) + ", Precision " + str(
            #            batch_precision[0]) + ", Recall " + str(batch_recall[0])
            #          + ", F1_score" + str(batch_f1_score[0]))
            #batch_id = batch_id + 1
    '''
        pass_precision, pass_recall, pass_f1_score = chunk_evaluator.eval(exe)
        print("[TrainSet] pass_id:" + str(pass_id) + " pass_precision:" + str(
            pass_precision) + " pass_recall:" + str(pass_recall) +
              " pass_f1_score:" + str(pass_f1_score))
        pass_precision, pass_recall, pass_f1_score = test(
            exe, chunk_evaluator, inference_program, test_reader, place)
        print("[TestSet] pass_id:" + str(pass_id) + " pass_precision:" + str(
            pass_precision) + " pass_recall:" + str(pass_recall) +
              " pass_f1_score:" + str(pass_f1_score))

        save_dirname = os.path.join(model_save_dir, "params_pass_%d" % pass_id)
        fluid.io.save_inference_model(save_dirname, ['word', 'mention', 'target'],
                                      [crf_decode], exe)
    '''

if __name__ == "__main__":
    main(
        train_data_file="data/train.txt",
        test_data_file="data/test.txt",
        vocab_file="data/word_dict",
        target_file="data/label_dict",
        model_save_dir="models",
        num_passes=2000,
        use_gpu=True,
        parallel=True)
