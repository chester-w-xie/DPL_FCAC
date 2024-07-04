"""
-------------------------------File info-------------------------
% - File name: Meta_trian_FSC89.py
% - Description:
% -
% -
% - Input:
% - Output:  None
% - Calls: None
% - usage:
% - Version： V1.0
% - Last update: 2022-09-02
%  Copyright (C) PRMI, South China university of technology; 2022
%  ------For Educational and Academic Purposes Only ------
% - Author : Chester.Wei.Xie, PRMI, SCUT/ GXU
% - Contact: ee_w.xie@mail.scut.edu.cn
------------------------------------------------------------------
"""
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F
from tqdm import tqdm
from DatasetsManager_FSC89 import fsc89_dataset_for_fscil, fsc89_dataset_for_fscil_augmix
from torchsummary import summary
from utils import *
import math
from torch.utils.data import DataLoader
import logging
import sys
import argparse

from Meta_model_define import MYNET, replace_base_fc
from results_assemble import get_results_assemble


class Trainer(object):
    def __init__(self, args):

        self.scheduler = None
        self.args = args

        self.datasets = fsc89_dataset_for_fscil(args)
        self.datasets_augmix = fsc89_dataset_for_fscil_augmix(args)
        self.label_per_task = [list(np.array(range(args.base_class)))] + [list(np.array(range(args.way)) +
                                                                               args.way * task_id + args.base_class)
                                                                          for task_id in range(args.tasks)]
        self.base_class_num = args.base_class
        self.test_results_one_trial = {}
        self.test_results_all_trial = {}
        self.num_sessions = args.session
        self.pretrain_model_dir = os.path.join(args.pretrained_model_path,
                                               'pretrained_model_' + args.dataset_name + '.pth')

        self.model = MYNET(self.args, mode=self.args.base_mode)

        if os.path.isfile(self.pretrain_model_dir):
            logging.info('loading pretrained model form %s\n', self.pretrain_model_dir)
            para = torch.load(self.pretrain_model_dir)
            self.model = update_param(self.model, para)

            self.model = self.model.cuda()
        else:
            logging.info('random init params\n')
            self.model = self.model.cuda()
        if args.start_session > 0:
            logging.info('WARING: Random init weights for new sessions!\n')

        self.best_model_dict = deepcopy(self.model.state_dict())

        self.best_pred = 0.0
        self.val_loss_min = None
        self.best_result_dic = {}
        self.early_stopping_count = 0
        # history of prediction
        self.acc_history = []
        self.best_result_dir = os.path.join(args.dir_name, 'session_0_bset_result_' + args.dataset_name + '.pth')

    def fit(self):
        logging.info('meta training...\n')
        self.meta_training()
        logging.info('meta training is done.\n')

        logging.info('Start meta testing...\n')
        meta_test_start_time = time.time()
        for trial in range(self.args.trials):

            meta_model = MYNET(self.args, mode=self.args.base_mode)
            para = torch.load(self.best_result_dir)['model']

            meta_model = meta_model.cuda()

            meta_model = update_param(meta_model, para)
            logging.info('Meta testing (Support set: %d way %d shot):' % (self.args.way, self.args.shot))
            for session in range(1, self.num_sessions):
                updated_model = self.meta_testing(session, trial, meta_model)
                meta_model = updated_model

            self.test_results_all_trial[trial] = self.test_results_one_trial.copy()
        meta_test_end_time = time.time()

        meta_spend_time = (meta_test_end_time - meta_test_start_time) / self.args.trials
        meta_spend_time = meta_spend_time / (self.num_sessions - 1)

        avg_meta_test_time = format_time(meta_spend_time)
        logging.info('meta-testing is done! avg running time (raw) over sessions is {:8}.\n'.format(meta_spend_time))
        logging.info(
            'meta-testing is done! avg running time (format) over sessions is {:8}.\n'.format(avg_meta_test_time))

        results_save_path = os.path.join(self.args.dir_name, 'test_results_{}_trial.pth'.format(self.args.trials))
        torch.save(self.test_results_all_trial, results_save_path)
        print(f'All results have been saved to {results_save_path}')
        get_results_assemble(results_save_path)

    def meta_training(self, current_session=0, current_trial=1):
        train_dataset_raw = self.datasets['train'][current_session]
        train_dataset = self.datasets_augmix['train'][current_session]

        val_dataset = self.datasets['val']
        session_class = self.args.base_class + self.args.way * current_session
        epochs = self.args.epochs_base

        sampler = CategoriesSamplerCEC(np.array(train_dataset.targets), self.args.train_episode,
                                       self.args.episode_way,
                                       self.args.episode_shot + self.args.episode_query)

        train_loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=4,
                                  pin_memory=True)

        optimizer = torch.optim.SGD([{'params': self.model.encoder.parameters(), 'lr': self.args.lr_base},
                                     {'params': self.model.Selective_attn.parameters(), 'lr': self.args.lrg}],
                                    momentum=0.9, nesterov=True, weight_decay=self.args.decay)

        scheduler = None
        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)

        for epoch in range(args.epochs_base):
            start_time = time.time()
            # train base sess
            self.model.eval()
            train_loss = 0.0
            num_iter = len(train_loader)

            tqdm_gen = tqdm(train_loader)

            label = torch.arange(args.episode_way + args.low_way).repeat(args.episode_query)
            label = label.type(torch.cuda.LongTensor)

            for i, (batch, true_label) in enumerate(tqdm_gen, 1):
                data, data_augmix1, data_augmix2 = [_.cuda() for _ in batch]

                k = args.episode_way * args.episode_shot

                proto_tmp, query_tmp = data_augmix1[:k], data_augmix1[k:]

                proto_tmp = proto_tmp.cuda()
                query_tmp = query_tmp.cuda()

                self.model.mode = 'encoder'
                data = self.model(data)
                proto_tmp = self.model(proto_tmp)
                query_tmp = self.model(query_tmp)

                proto, query = data[:k], data[k:]

                proto = proto.view(args.episode_shot, args.episode_way, proto.shape[-1])
                query = query.view(args.episode_query, args.episode_way, query.shape[-1])

                proto_tmp = proto_tmp.view(args.low_shot, args.low_way, proto.shape[-1])
                query_tmp = query_tmp.view(args.episode_query, args.low_way, query.shape[-1])

                proto = proto.mean(0).unsqueeze(0)
                proto_tmp = proto_tmp.mean(0).unsqueeze(0)

                base_proto = proto.clone()  # [1, 15,64]
                base_proto = base_proto.unsqueeze(0)  # [1,1,15,64]

                proto = torch.cat([proto, proto_tmp], dim=1)
                query = torch.cat([query, query_tmp], dim=1)

                proto = proto.unsqueeze(0)
                query = query.unsqueeze(0)

                logits = self.model._forward(proto, query, base_proto)

                loss = F.cross_entropy(logits, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                acc = count_acc(logits, label)

                train_loss += loss.item()
            self.model = replace_base_fc(train_dataset_raw, self.model, args)

            self.model.mode = 'avg_cos'

            val_loss = self.validation(val_dataset)
            self.keep_record_of_best_model(val_loss, epoch)

            logging.info('[Meta training, Epoch: {}/{},'
                         ' num. of training samples: {}.'
                         ' ==> training loss: {:.3f},'
                         ' , val loss: {:.3f}]\n'.format(epoch + 1, epochs,
                                                         (num_iter - 1) * self.args.batch_size +
                                                         data.data.shape[0],
                                                         train_loss / num_iter, val_loss)
                         )
            scheduler.step()

        self.model = replace_base_fc(train_dataset_raw, self.model, args)

        logging.info('Replace the fc with average embedding, and save it to :%s \n' % self.best_result_dir)
        self.best_model_dict = deepcopy(self.model.state_dict())
        # undate result dic
        self.best_result_dic = {'model': self.best_model_dict}

        torch.save(self.best_result_dic, self.best_result_dir)

        self.model.mode = 'avg_cos'

        val_loss = self.validation(val_dataset)
        logging.info('The new best val loss of base session={:.3f}'.format(val_loss))

        self.evaluate(current_session, current_trial, self.model)

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

            self.model.mode = 'encoder'
            query = self.model(sample)
            query = query.unsqueeze(0).unsqueeze(0)

            proto = self.model.fc.weight[:session_class, :].detach()
            proto = proto.unsqueeze(0).unsqueeze(0)

            with torch.no_grad():

                batch_output = self.model._forward(proto, query, proto)
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

            eval_model.mode = 'encoder'
            query = eval_model(data)
            query = query.unsqueeze(0).unsqueeze(0)

            proto = eval_model.fc.weight[:session_class, :].detach()
            base_proto = eval_model.fc.weight[:self.args.base_class, :].detach()
            proto = proto.unsqueeze(0).unsqueeze(0)
            base_proto = base_proto.unsqueeze(0).unsqueeze(0)

            with torch.no_grad():

                batch_output = eval_model._forward(proto, query, base_proto)
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
    parser = argparse.ArgumentParser(description='Meta train for FSC89')

    # about dataset and network
    parser.add_argument('-project', type=str, default='DPL')
    parser.add_argument('-dataset', type=str, default='FSC89_mini',
                        choices=['FSC89_mini', 'FSC89_huge'])
    parser.add_argument('-dataroot', type=str, default='data/')
    parser.add_argument('--fcac_method', type=str, default='DPL', help='fcac method (default: None)')

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

    parser.add_argument('-batch_size', type=int, default=225)

    parser.add_argument('-test_batch_size', type=int, default=100)
    parser.add_argument('-base_mode', type=str, default='ft_cos',
                        choices=['ft_dot', 'ft_cos'])

    parser.add_argument('-new_mode', type=str, default='avg_cos',
                        choices=['ft_dot', 'ft_cos', 'avg_cos'])

    # for episode learning
    parser.add_argument('-train_episode', type=int, default=100)
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
    parser.add_argument('-gpu', default='0,1')
    parser.add_argument('-num_workers', type=int, default=8)
    parser.add_argument('-seed', type=int, default=1)
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

    # AugMix options
    parser.add_argument('--mixture-width', default=3, type=int, help='Number of augmentation chains to '
                                                                     'mix per augmented example')
    parser.add_argument('--mixture-depth', default=-1, type=int,
                        help='Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]')
    parser.add_argument('--aug-severity', default=3, type=int, help='Severity of base augmentation operators')
    parser.add_argument('--no-jsd', '-nj', action='store_true', help='Turn off JSD consistency loss.')
    parser.add_argument('--all-ops', '-all', action='store_true', help='Turn on all operations '
                                                                       '(+brightness,contrast,color,sharpness).')

    _args = parser.parse_args()
    return _args


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
    pprint(vars(args))
    args.num_gpu = set_gpu(args)
    args.tasks = args.session - 1

    args.all_class = args.base_class + args.way * args.tasks
    args.now_time = str(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))

    args.dir_name = 'exp/' + str(args.dataset_name) + '-' + str(args.num_class) + '_' + str(
        args.fcac_method) + '_' \
                    + str(args.way) + 'way' + '_' + str(args.shot) + 'shot' + '_' + str(args.now_time)
    args.pretrained_model_path = 'exp/' + str(args.dataset_name) + '-' \
                                 + str(args.way) + 'way' + '_' + str(args.shot) + 'shot' + '_' + 'Pretrain_model'
    print(f'check: {args.pretrained_model_path}')
    if not os.path.exists(args.dir_name):
        os.makedirs(args.dir_name)

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
