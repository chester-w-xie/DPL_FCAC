"""
-------------------------------File info-------------------------
% - File name: Pre_train_FSC89.py
% - Description:
% -
% - Input:
% - Output:  None
% - Calls: None
% - usage:
% - Version： V1.0
% - Last update: 2022-08-30
%  Copyright (C) PRMI, South China university of technology; 2022
%  ------For Educational and Academic Purposes Only ------
% - Author : Chester.Wei.Xie, PRMI, SCUT/ GXU
% - Contact: chester.w.xie@gmail.com
------------------------------------------------------------------
"""

import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F
from tqdm import tqdm
from DatasetsManager_FSC89 import fsc89_dataset_for_fscil
from torchsummary import summary
from utils import *
import math
from torch.utils.data import DataLoader
import logging
import sys
import argparse
from Base_model_define import FscilModel, replace_base_fc
from results_assemble import get_results_assemble


class Trainer(object):
    def __init__(self, args):

        self.scheduler = None
        self.args = args

        self.datasets = fsc89_dataset_for_fscil(args)
        self.label_per_task = [list(np.array(range(args.base_class)))] + [list(np.array(range(args.way)) +
                                                                               args.way * task_id + args.base_class)
                                                                          for task_id in range(args.tasks)]
        self.base_class_num = args.base_class
        self.test_results_one_trial = {}
        self.test_results_all_trial = {}
        self.num_sessions = args.session

        # Define model and optimizer
        self.model = FscilModel(self.args, mode=self.args.base_mode)

        self.model = self.model.cuda()

        print('random init params')
        if args.start_session > 0:
            print('WARING: Random init weights for new sessions!')

        self.best_model_dict = deepcopy(self.model.state_dict())

        self.best_pred = 0.0
        self.val_loss_min = None
        self.best_result_dic = {}
        self.early_stopping_count = 0
        # history of prediction
        self.acc_history = []
        self.best_result_dir = os.path.join(args.dir_name, 'pretrained_bset_result_' + args.dataset_name + '.pth')
        self.pretrain_model_dir = os.path.join(args.pretrained_model_path,
                                               'pretrained_model_' + args.dataset_name + '.pth')

    def fit(self):
        # pretraining
        logging.info('pretraining the model...\n')
        self.pretraining()
        logging.info('pretraining is done.\n')

        logging.info('Start meta testing...\n')
        for trial in range(self.args.trials):

            meta_model = FscilModel(self.args, mode=self.args.base_mode)
            para = torch.load(self.best_result_dir)['model']

            meta_model = meta_model.cuda()

            meta_model = update_param(meta_model, para)
            logging.info('Meta testing (Support set: %d way %d shot):' % (self.args.way, self.args.shot))
            for session in range(1, self.num_sessions):
                updated_model = self.meta_testing(session, trial, meta_model)
                meta_model = updated_model
            self.test_results_all_trial[trial] = self.test_results_one_trial.copy()

        results_save_path = os.path.join(self.args.dir_name, 'test_results_{}_trial.pth'.format(self.args.trials))
        torch.save(self.test_results_all_trial, results_save_path)
        print(f'All results have been saved to {results_save_path}')
        get_results_assemble(results_save_path)

    def pretraining(self, current_session=0, current_trial=1):
        train_dataset = self.datasets['train'][current_session]
        val_dataset = self.datasets['val']
        session_class = self.args.base_class + self.args.way * current_session
        epochs = self.args.epochs_base

        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=4,
                                  pin_memory=True)

        self.model.load_state_dict(self.best_model_dict)
        #
        optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr_base, momentum=0.9, nesterov=True,
                                    weight_decay=self.args.decay)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                         gamma=self.args.gamma)

        for epoch in range(self.args.epochs_base):
            start_time = time.time()

            train_loss = 0.0
            num_iter = len(train_loader)
            tbar = tqdm(train_loader)
            self.model.train()

            # standard classification for pretrain

            for i, batch in enumerate(tbar):
                data, train_label = [_.cuda() for _ in batch]

                logits = self.model(data)
                logits = logits[:, :self.args.base_class]
                loss = F.cross_entropy(logits, train_label)
                acc = count_acc(logits, train_label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            val_loss = self.validation(val_dataset)
            self.keep_record_of_best_model(val_loss, epoch)

            logging.info('[Pretraining, Epoch: {}/{},'
                         ' num. of training samples: {}.'
                         ' ==> training loss: {:.3f}'
                         ' , val loss: {:.3f}]\n'.format(epoch + 1, epochs,
                                                         (num_iter - 1) * self.args.batch_size +
                                                         data.data.shape[0],
                                                         train_loss / num_iter, val_loss)
                         )
            scheduler.step()

        if not args.not_data_init:

            self.model.load_state_dict(self.best_model_dict)

            self.model = replace_base_fc(train_dataset, self.model, args)

            logging.info('Replace the fc with average embedding, and save it to :%s \n' % self.best_result_dir)
            self.best_model_dict = deepcopy(self.model.state_dict())
            # undate result dic
            self.best_result_dic = {'model': self.best_model_dict}

            torch.save(self.best_result_dic, self.best_result_dir)

            self.model.mode = 'avg_cos'

            val_loss = self.validation(val_dataset)
            logging.info('The new best val loss of base session={:.3f}'.format(val_loss))

            self.evaluate(current_session, current_trial, self.model)

        torch.save(self.best_model_dict, self.pretrain_model_dir)
        logging.info('meta-training is done, the best model is saving to %s \n' % self.pretrain_model_dir)

    def validation(self, dataset):
        self.model.eval()

        val_loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=4)
        vbar = tqdm(val_loader)
        session_class = self.args.base_class

        outputs = []
        targets = []
        for i, batch_samples in enumerate(vbar):
            sample, target = batch_samples[0], batch_samples[1]
            targets.append(target)
            sample = sample.cuda()

            with torch.no_grad():
                batch_output = self.model(sample)[:, :session_class]
                outputs.append(batch_output.data.cpu().numpy())

        outputs = np.concatenate(outputs, axis=0)
        targets = np.concatenate(targets, axis=0)
        val_loss = float(F.cross_entropy(torch.Tensor(outputs), torch.LongTensor(targets)).numpy())

        return val_loss

    def keep_record_of_best_model(self, val_loss, epoch):
        self.early_stopping_count += 1
        if self.val_loss_min is None or val_loss < self.val_loss_min:
            logging.info('Update best model and reset counting.')

            self.early_stopping_count = 0
            self.val_loss_min = val_loss
            # undate result dic
            self.best_result_dic = {'val_loss': val_loss,
                                    'model': self.model.state_dict(),
                                    'epoch': epoch
                                    }

            self.best_model_dict = deepcopy(self.model.state_dict())

    def meta_testing(self, current_session, current_trial, _trained_model):

        meta_test_datasets = fsc89_dataset_for_fscil(self.args)

        meta_loader = DataLoader(meta_test_datasets['train'][current_session], batch_size=2048,
                                 shuffle=False, num_workers=4,
                                 pin_memory=True)
        train_set = meta_test_datasets['train'][current_session]

        _trained_model.mode = self.args.new_mode
        _trained_model.eval()

        _trained_model.update_fc(meta_loader, np.unique(list(train_set.sub_indexes.keys())), current_session)
        self.evaluate(current_session, current_trial, _trained_model)

        return _trained_model

    def evaluate(self, current_session, current_trial, trained_model):

        eval_model = trained_model
        eval_model.eval()

        test_dataset = self.datasets['test'][current_session]
        test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=4)
        session_class = self.args.base_class + self.args.way * current_session

        outputs = []
        targets = []
        for i, batch in enumerate(test_loader):
            data, target = batch
            data = data.cuda()
            targets.append(target)

            with torch.no_grad():
                batch_output = eval_model(data)[:, :session_class]
                outputs.append(batch_output.data.cpu().numpy())

        outputs = np.concatenate(outputs, axis=0)
        targets = np.concatenate(targets, axis=0)

        audio_predictions = np.argmax(outputs, axis=-1)  # (audios_num,)
        # Evaluate
        classes_num = outputs.shape[-1]

        test_set_acc_overall = calculate_accuracy(targets, audio_predictions,
                                                  classes_num, average='macro')
        class_wise_acc = calculate_accuracy(targets, audio_predictions, classes_num)
        cf_matrix = calculate_confusion_matrix(targets, audio_predictions, classes_num)

        class_wise_acc_base = class_wise_acc[:self.base_class_num]

        class_wise_acc_all_novel = class_wise_acc[self.base_class_num:]
        #
        class_wise_acc_previous_novel = class_wise_acc[self.base_class_num:(self.base_class_num + self.args.way)]

        class_wise_acc_current_novel = class_wise_acc[-self.args.way:]

        # Test
        logging.info('[Trial: %d, Session: %d, num. of seen classes: %d,'
                     ' num. test samples: %5d]' % (current_trial, current_session,
                                                   session_class, i * self.args.batch_size + data.data.shape[0]))

        if current_session == 0:
            logging.info("==> Average of class wise acc: {:.2f} (base)"
                         ", - (all novel)"
                         ", - (previous novel)"
                         ", - (current novel)"
                         ", {:.2f} (both)\n".format(np.mean(class_wise_acc_base) * 100,
                                                    test_set_acc_overall * 100)
                         )

            ave_acc_all_novel = None
            ave_acc_previous_novel = None
            ave_acc_current_novel = None
        else:

            ave_acc_all_novel = np.mean(class_wise_acc_all_novel)
            ave_acc_previous_novel = np.mean(class_wise_acc_previous_novel)
            ave_acc_current_novel = np.mean(class_wise_acc_current_novel)

            logging.info("==> Average of class wise acc: {:.2f} (base)"
                         ", {:.2f} (all novel)"
                         ", {:.2f} (previous novel)"
                         ", {:.2f} (current novel)"
                         ", {:.2f} (both)\n".format(np.mean(class_wise_acc_base) * 100,
                                                    ave_acc_all_novel * 100,
                                                    ave_acc_previous_novel * 100,
                                                    ave_acc_current_novel * 100,
                                                    test_set_acc_overall * 100)
                         )

        session_results_dict = {'Ave_class_wise_acc_base': np.mean(class_wise_acc_base),
                                'Ave_class_wise_acc_all_novel': ave_acc_all_novel,
                                'Ave_class_wise_acc_previous_novel': ave_acc_previous_novel,
                                'Ave_class_wise_acc_current_novel': ave_acc_current_novel,
                                'Ave_acc_of_both': test_set_acc_overall,
                                }
        self.test_results_one_trial[current_session] = session_results_dict.copy()

        if current_session == self.num_sessions - 1:
            self.show_results_summary(current_trial)

    def show_results_summary(self, current_trial):

        base_avg_over_sessions = []
        all_avg_novel_over_sessions = []
        pre_avg_novel_over_sessoins = []
        curr_avg_novel_over_sessions = []
        both_avg_over_sessions = []

        logging.info('=====> Trial {} results summary, '
                     '(Support set: {} way {} shot)'.format(current_trial, self.args.way, self.args.shot))
        print(f'-------------------- Average of class-wise acc (%)--------------------------------')
        print(f'\n Session         ', end=" ")
        for _, n in enumerate(self.test_results_one_trial.keys()):
            print(f'{n}', end="\t")
        print(f'Average', end="\t")

        print(f'\n Base      ', end="\t")
        for _, n in enumerate(self.test_results_one_trial.keys()):
            temp = self.test_results_one_trial[n]['Ave_class_wise_acc_base']
            print(f'{temp * 100:.2f}', end="\t")
            base_avg_over_sessions.append(temp)

        print(f'{np.mean(base_avg_over_sessions) * 100:.2f}', end="\t")

        print(f'\n All Novel       ', end=" ")
        for _, n in enumerate(self.test_results_one_trial.keys()):

            if n == 0:
                print(f'-', end="\t")
            else:
                temp = self.test_results_one_trial[n]['Ave_class_wise_acc_all_novel']
                print(f'{temp * 100:.2f}', end="\t")
                all_avg_novel_over_sessions.append(temp)
        print(f'{np.mean(all_avg_novel_over_sessions) * 100:.2f}', end="\t")
        print(f'\n Previous Novel  ', end=" ")
        for _, n in enumerate(self.test_results_one_trial.keys()):

            if n == 0:
                print(f'-', end="\t")
            else:
                temp = self.test_results_one_trial[n]['Ave_class_wise_acc_previous_novel']
                print(f'{temp * 100:.2f}', end="\t")
                pre_avg_novel_over_sessoins.append(temp)

        print(f'{np.mean(pre_avg_novel_over_sessoins) * 100:.2f}', end="\t")

        print(f'\n Current Novel   ', end=" ")
        for _, n in enumerate(self.test_results_one_trial.keys()):

            if n == 0:
                print(f'-', end="\t")
            else:
                temp = self.test_results_one_trial[n]['Ave_class_wise_acc_current_novel']
                print(f'{temp * 100:.2f}', end="\t")
                curr_avg_novel_over_sessions.append(temp)
        print(f'{np.mean(curr_avg_novel_over_sessions) * 100:.2f}', end="\t")

        print(f'\n Both     ', end="\t")
        for _, n in enumerate(self.test_results_one_trial.keys()):
            temp = self.test_results_one_trial[n]['Ave_acc_of_both']
            print(f'{temp * 100:.2f}', end="\t")
            both_avg_over_sessions.append(temp)
        print(f'{np.mean(both_avg_over_sessions) * 100:.2f}', end="\t")
        print(f'\n --------------------------------------------------------------------------------\n ')

        PD = self.test_results_one_trial[0]['Ave_acc_of_both'] - \
             self.test_results_one_trial[self.num_sessions - 1]['Ave_acc_of_both']

        temp2 = self.test_results_one_trial[0]['Ave_class_wise_acc_base'] - \
                self.test_results_one_trial[self.num_sessions - 1]['Ave_class_wise_acc_base']
        # FR->  forgetting rate , MR-> memorizing rate  for all sessions
        FR_overall = temp2 / self.test_results_one_trial[0]['Ave_class_wise_acc_base']
        FR_overall_avg = FR_overall / (self.num_sessions - 1)
        MR_overall = 1 - FR_overall

        # FR,MR average over all sessions
        FR_session_list = []
        FR_session_list_temp = []
        for _session in range(1, self.num_sessions):
            acc_previous = self.test_results_one_trial[_session - 1]['Ave_class_wise_acc_base']
            acc_current = self.test_results_one_trial[_session]['Ave_class_wise_acc_base']
            FR_session = (acc_previous - acc_current) / acc_previous
            FR_session_list.append(FR_session)
            FR_session_list_temp.append(FR_session * 100)
        FR_session_avg = np.mean(FR_session_list)
        MSR_session_avg = 1 - FR_session_avg

        # CPS = 0.5 * MSR_session_avg + 0.5 * np.mean(all_avg_novel_over_sessions)
        CPS = 0.5 * MR_overall + 0.5 * np.mean(all_avg_novel_over_sessions)

        logging.info(' ==> PD: {:.2f} (define by CEC); \n'.format(PD * 100))

        logging.info(' =====> FR_overall: {:.2f}, FR_overall_avg: {:.2f},'
                     ' MR_overall: {:.2f}; \n'.format(FR_overall * 100, FR_overall_avg * 100, MR_overall * 100))

        logging.info(
            '  =====> FR_session_avg: {:.2f}, '
            'MSR_session_avg: {:.2f};'.format(FR_session_avg * 100, MSR_session_avg * 100))

        logging.info('  =====> Average of all novel acc over {} incremental sessions: {:.2f};'.format(
            self.num_sessions - 1, np.mean(all_avg_novel_over_sessions) * 100))
        # logging.info('  =====> CPS: {:.2f} \n'.format(CPS * 100))
        logging.info('  =====> CPS: {:.2f} \n'.format(CPS * 100))


def setup_parser():
    parser = argparse.ArgumentParser(description='FCAC for FSC89')

    # about dataset and network
    parser.add_argument('-project', type=str, default='Pretrain')

    parser.add_argument('--fcac_method', type=str, default='Pretrain', help='fcac method (default: None)')
    parser.add_argument('--do_norm', action='store_true', help='norm the features')
    parser.add_argument('--im_pretrain', action='store_true', help='Load pre-trained parameters')

    # about pre-training
    parser.add_argument('-epochs_base', type=int, default=100)
    parser.add_argument('-epochs_new', type=int, default=100)
    parser.add_argument('-lr_base', type=float, default=0.1)
    parser.add_argument('-lr_new', type=float, default=0.1)
    parser.add_argument('-schedule', type=str, default='Step',
                        choices=['Step', 'Milestone'])
    parser.add_argument('-milestones', nargs='+', type=int, default=[60, 70])
    parser.add_argument('-step', type=int, default=40)
    parser.add_argument('-decay', type=float, default=0.0005)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-gamma', type=float, default=0.1)
    parser.add_argument('-temperature', type=int, default=16)
    parser.add_argument('-not_data_init', action='store_true', help='using average data embedding to init or not')

    parser.add_argument('-batch_size', type=int, default=128)

    parser.add_argument('-test_batch_size', type=int, default=100)
    parser.add_argument('-base_mode', type=str, default='ft_cos',
                        choices=['ft_dot', 'ft_cos'])
    # ft_dot means using linear classifier, ft_cos means using cosine classifier
    parser.add_argument('-new_mode', type=str, default='avg_cos',
                        choices=['ft_dot', 'ft_cos', 'avg_cos'])
    # ft_dot means using linear classifier, ft_cos means using cosine classifier, avg_cos means
    # using average data embedding and cosine classifier

    # for episode learning
    parser.add_argument('-train_episode', type=int, default=50)
    parser.add_argument('-episode_shot', type=int, default=1)
    parser.add_argument('-episode_way', type=int, default=15)
    parser.add_argument('-episode_query', type=int, default=15)

    parser.add_argument('-lrg', type=float, default=0.1)
    parser.add_argument('-low_shot', type=int, default=1)
    parser.add_argument('-low_way', type=int, default=15)

    parser.add_argument('-start_session', type=int, default=0)
    parser.add_argument('-model_dir', type=str, default=None, help='loading model parameter from a specific dir')
    parser.add_argument('-set_no_val', action='store_true', help='set validation using test set or no validation')

    # about training
    parser.add_argument('-gpu', default='0')
    parser.add_argument('-num_workers', type=int, default=8)
    parser.add_argument('-seed', type=int, default=1668)
    parser.add_argument('-debug', action='store_true')

    # dir
    parser.add_argument('--metapath', type=str, required=True, help='path to FSC-89-meta folder')
    parser.add_argument('--datapath', type=str, required=True, help='path to FSD-MIX-CLIPS_data folder)')
    parser.add_argument('--setup', type=str, required=True, help='mini or huge')
    parser.add_argument('--data_type', type=str, required=True, help='audio or openl3)')
    parser.add_argument('--num_class', type=int, default=89, help='Total number of classes in the dataset')

    # dataset option
    parser.add_argument('--dataset_name', type=str, default='FSC89_mini',
                        help='dataset name (default: FSC89_mini)')

    # dataset setting(class-division, way, shot)
    parser.add_argument('--base_class', type=int, default=59, help='number of base class (default: 60)')
    parser.add_argument('--way', type=int, default=5, help='class number of per task (default: 5)')
    parser.add_argument('--shot', type=int, default=5, help='shot of per class (default: 5)')
    parser.add_argument('--base_start_index', type=int, default=0, help='start label index for base class (default: 0)')

    # hyper option
    parser.add_argument('--session', type=int, default=7, metavar='N',
                        help='num. of sessions, including one base session and n incremental sessions (default:10)')
    parser.add_argument('--trials', type=int, default=100, metavar='N',
                        help='num. of trials for the incremental sessions (default:100)')
    parser.add_argument('--early_stop_tol', type=int, default=10, metavar='N',
                        help='tolerance for early stopping (default:10)')

    _args = parser.parse_args()
    return _args


def set_device(args_):
    # if args.cudnn:
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.enabled = True
    #
    # torch.manual_seed(args_.seed)
    # torch.cuda.manual_seed(args_.seed)
    # np.random.seed(args_.seed)
    # random.seed(args_.seed)

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args_.gpu_id


def update_param(model, pretrained_dict):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items()}
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


if __name__ == "__main__":

    args = setup_parser()
    set_seed(args.seed)
    # set_device(args)
    pprint(vars(args))
    args.num_gpu = set_gpu(args)
    args.tasks = args.session - 1

    args.all_class = args.base_class + args.way * args.tasks
    args.now_time = str(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))

    args.dir_name = 'exp/' + str(args.dataset_name) + '-' + str(args.fcac_method) + '_' \
                    + str(args.way) + 'way' + '_' + str(args.shot) + 'shot' + '_' + str(args.now_time)
    args.pretrained_model_path = 'exp/' + str(args.dataset_name) + '-' \
                                 + str(args.way) + 'way' + '_' + str(args.shot) + 'shot' + '_' + 'Pretrain_model'

    if not os.path.exists(args.dir_name):
        os.makedirs(args.dir_name)

    if not os.path.exists(args.pretrained_model_path):
        os.makedirs(args.pretrained_model_path)

    logging.basicConfig(level=logging.INFO,
                        filename=args.dir_name + '/output_logging_' + args.now_time + '.log',
                        datefmt='%Y/%m/%d %H:%M:%S',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info('\nAll args of the experiment ====>')
    logging.info(args)
    logging.info('\n\n')

    start_time = time.time()
    trainer = Trainer(args)
    trainer.fit()
    end_time = time.time()

    time_spent = format_time(end_time - start_time)
    logging.info('All done! The entire process took {:8}.\n'.format(time_spent))
