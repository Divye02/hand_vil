import numpy as np
from settings import *
# import torch
from mjrl.utils import tensor_utils
# import mjrl.envs
from mjrl.utils.get_environment import get_environment
from tqdm import tqdm
from data_gen.dataset import *
from data_gen.transforms import *
import os
from settings import FRAME_SIZE
import pickle
from torch.utils.data import DataLoader
from training.logger import Logger


class Dagger:
    def __init__(self,
                 dagger_epochs,
                 expert_policy,
                 viz_policy,
                 old_data_loader: DataLoader,
                 val_data_loader: DataLoader,
                 log_dir,
                 pol_dir_name,
                 has_robot_info=False,
                 beta_start=1.0,
                 beta_decay=0.9,
                 beta_cutoff=0.0,
                 optimizer=None,
                 camera_name=None,
                 lr=3e-4,
                 log_step=10,
                 bins=0,
                 use_img=True,
                 use_seq=True,
                 trainer_epochs=5,
                 num_traj_gen=20,
                 env_name=None,
                 env=None,
                 save_epoch=1,
                 eval_num_traj=25,
                 seed=500,
                 sliding_window=0,
                 device_id=None,
                 use_cuda=False):

        self.beta = beta_start
        self.dagger_epochs = dagger_epochs
        self.expert_policy = expert_policy
        self.viz_policy = viz_policy
        self.old_data_loader = old_data_loader
        self.beta_decay = beta_decay
        self.camera_name = camera_name
        self.has_robot_info = has_robot_info
        self.beta_cutoff = beta_cutoff
        self.log_step = log_step
        self.bins = bins
        self.pol_dir_name = pol_dir_name
        self.use_img = use_img
        self.use_seq = use_seq
        self.trainer_epochs = trainer_epochs
        self.val_data_loader = val_data_loader
        self.num_traj_gen = num_traj_gen
        self.eval_num_traj = eval_num_traj
        self.env = env
        self.env_name = env_name
        self.save_epoch = save_epoch
        self.sliding_window = sliding_window
        self.device_id = device_id
        self.use_cuda = use_cuda

        # filewriters
        self.log_tf_train = Logger(os.path.join(LOG_DIR, log_dir))
        self.log_tf_val = Logger(os.path.join(LOG_DIR, log_dir, 'validation'))
        self.log_expert = Logger(os.path.join(LOG_DIR, log_dir, 'expert'))
        self.log_viz = Logger(os.path.join(LOG_DIR, log_dir, 'viz'))

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.viz_policy.trainable_params, lr=lr) if optimizer is None else optimizer

        self.seed = seed
        if self.env_name is None and self.env is None:
            print("No environment specified! Error will be raised")
        if self.env is None: self.env = get_environment(self.env_name)

        self.expert_reward, _, _ = self.env.evaluate_policy(self.expert_policy,
                                                            num_episodes=self.eval_num_traj, mean_action=True,
                                                            seed=self.seed, device_id=self.device_id)

    def train(self):
        beta = self.beta
        step = 0
        print('-' * 80)
        print('Reward evaluation')
        # dagger reward evaluation
        self.plot_reward(0)
        for dagger_epoch in tqdm(range(self.dagger_epochs)):
            print('=' * 80)
            print('Dagger Epoch: %i (Beta: %f)' % (dagger_epoch, beta))

            # collect some data using the actor and critic policies
            print('-' * 80)
            print('Generating Trajectories')
            paths = self.trajectory_generator(beta, dagger_epoch)

            # get a loader for the new+old data
            print('-' * 80)
            print('Loading Data')
            dataloader = self.dataloader_generator(paths)

            # train on the data
            print('-' * 80)
            print('Training')
            step = self.trainer(dataloader, step)

            # save model after each save_epoch
            if (dagger_epoch + 1) % self.save_epoch == 0:
                ensure_dir(self.pol_dir_name)
                self.viz_policy.cpu()
                pickle.dump(self.viz_policy,
                            open(os.path.join(self.pol_dir_name, 'trained_policy_ep_%d.pickle' % (dagger_epoch + 1)),
                                 'wb'))
                if self.use_cuda:
                    self.viz_policy.cuda()

            self.old_data_loader = dataloader

            beta *= self.beta_decay
            if beta <= self.beta_cutoff:
                beta = 0

            print('-' * 80)
            print('Reward evaluation')
            # dagger reward evaluation
            self.plot_reward(dagger_epoch + 1)

        if self.env is not None:
            del (self.env)

    def loss(self, obs, act, bins=0, robot_info=None):
        if bins != 0:
            act_pred = self.viz_policy.model.forward(obs.float(), robot_info=robot_info)
            act_pred = act_pred.view(obs.shape[0], -1, bins)
            num_actions = act.shape[1]
            loss = sum([self.loss_fn(act_pred[:, i, :], act[:, i]) for i in range(num_actions)])
            return loss
        else:
            LL, mu, log_std = self.viz_policy.new_dist_info(obs, act, robot_info=robot_info)
            # minimize negative log likelihood
            return -torch.mean(LL)

    def trainer(self, dataloader, step):
        for ep in range(self.trainer_epochs):
            # calculating validation loss
            count = 0
            total_loss = 0
            for batch_idx, sample_batched in enumerate(self.val_data_loader):
                self.optimizer.zero_grad()
                loss = self.get_loss(sample_batched).cpu().data
                total_loss += loss
                count += 1
            total_loss /= count
            self.log_tf_val.scalar_summary('loss', total_loss, step)

            # training step
            print('training trainer epoch %d' % ep)
            for batch_idx, sample_batched in tqdm(enumerate(dataloader)):
                self.optimizer.zero_grad()
                loss = self.get_loss(sample_batched)
                if step % self.log_step == 0:
                    self.log_tf_train.scalar_summary('loss', loss, step)
                step += 1
                loss.backward()
                self.optimizer.step()

        params_after_opt = self.viz_policy.get_param_values()
        self.viz_policy.set_param_values(params_after_opt, set_new=True, set_old=True)
        if self.use_cuda:
            self.viz_policy.model.cuda()
            self.viz_policy.old_model.cuda()
            self.viz_policy.log_std.data = self.viz_policy.log_std.data.cuda()
            self.viz_policy.old_log_std.data = self.viz_policy.old_log_std.data.cuda()
        return step

    def get_loss(self, sample_batched):
        key_x = 'observation' if not self.use_img else 'image' if not self.use_seq else 'image_seq'
        key_y = 'action_bin' if self.bins != 0 else 'action'

        obs = sample_batched[key_x]
        act = sample_batched[key_y]
        if self.use_cuda:
            obs = obs.cuda()
            act = act.cuda()
        robot_info = None
        if self.has_robot_info:
            robot_info = sample_batched['robot_info']
            if self.use_cuda:
                robot_info = robot_info.cuda()
        return self.loss(obs, act, self.bins, robot_info)

    def trajectory_generator(self, beta, dagger_ep):
        if self.env_name is None and self.env is None:
            print("No environment specified! Error will be raised")
        if self.env is None: self.env = get_environment(self.env_name)

        T = self.env.horizon

        paths = []
        print('Generating trajectories')
        for ep in tqdm(range(self.num_traj_gen)):
            if self.seed is not None:
                seed = self.seed + ep + dagger_ep * self.num_traj_gen
                self.env.env.env._seed(seed)
                np.random.seed(seed)
            else:
                np.random.seed()
            observations = []
            actions = []
            rewards = []
            agent_infos = []
            env_infos = []
            path_image_pixels = []
            all_robot_info = []

            o = self.env.reset()
            robot_info = None
            if self.has_robot_info:
                o, env_info = self.env.reset()
                robot_info = env_info['robot_info']
            done = False
            t = 0

            while t < T and done != True:
                r = np.random.random()
                image_pix = self.env.get_pixels(frame_size=FRAME_SIZE, camera_name=self.camera_name,
                                                device_id=self.device_id)

                a_expert, agent_info_expert = self.expert_policy.get_action(o)

                img = image_pix
                prev_img = image_pix
                prev_prev_img = image_pix
                if t > 0:
                    prev_img = path_image_pixels[t - 1]
                if t > 1:
                    prev_prev_img = path_image_pixels[t - 2]

                prev_prev_img = np.expand_dims(prev_prev_img, axis=0)
                prev_img = np.expand_dims(prev_img, axis=0)
                img = np.expand_dims(img, axis=0)

                o_img = np.concatenate((prev_prev_img, prev_img, img), axis=0)
                a_viz, agent_info_viz = self.viz_policy.get_action(o_img, use_seq=self.use_seq, use_cuda=self.use_cuda,
                                                                   robot_info=robot_info)

                if r <= beta:
                    a = agent_info_expert['evaluation']
                    agent_info = agent_info_expert
                else:
                    a = a_viz
                    agent_info = agent_info_viz

                next_o, r, done, env_info = self.env.step(a)
                observations.append(o)
                actions.append(agent_info_expert['evaluation'])
                rewards.append(r)
                agent_infos.append(agent_info)
                env_infos.append(env_info)
                path_image_pixels.append(image_pix)
                if self.has_robot_info:
                    all_robot_info.append(robot_info)
                    robot_info = env_info['robot_info']

                o = next_o
                t += 1

            path = dict(
                observations=np.array(observations),
                actions=np.array(actions),
                rewards=np.array(rewards),
                agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
                env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
                terminated=done,
                image_pixels=np.array(path_image_pixels)
            )
            if self.has_robot_info:
                path['robot_info'] = np.array(all_robot_info)

            paths.append(path)
        return paths

    def dataloader_generator(self, paths):
        old_dataset = self.old_data_loader.dataset
        old_data = old_dataset.get_paths(self.sliding_window)
        new_data = np.concatenate((old_data, paths))

        new_dataloader = DataLoader(get_dataset_from_paths(new_data, transforms=old_dataset.get_transform()),
                                    batch_size=self.old_data_loader.batch_size, shuffle=True, num_workers=4)
        return new_dataloader

    def plot_reward(self, step):
        # actions are that of expert still, but states/rewards are of viz_policy

        if self.use_cuda:
            self.viz_policy.cpu()
        reward, _, full_dist = self.env.evaluate_policy(self.viz_policy,
                                                        num_episodes=self.eval_num_traj, mean_action=True, use_seq=self.use_seq,
                                                        camera_name=self.camera_name, seed=self.seed,
                                                        get_full_dist=True, device_id=self.device_id)
        if self.use_cuda:
            self.viz_policy.cuda()

        # plot for mean, min, and max
        self.log_viz.histo_summary('Reward dist', full_dist, step, len(full_dist))
        rewards_dir = os.path.join(os.path.dirname(self.pol_dir_name), 'all_rewards')
        ensure_dir(rewards_dir)
        pickle.dump(full_dist, open(os.path.join(rewards_dir, 'full_rewards_ep_%d.pickle' % step), 'wb'))

        self.log_viz.scalar_summary('mean reward', reward[0], step)
        self.log_expert.scalar_summary('mean reward', self.expert_reward[0], step)
        self.log_viz.scalar_summary('min reward', reward[2], step)
        self.log_expert.scalar_summary('min reward', self.expert_reward[2], step)
        self.log_viz.scalar_summary('max reward', reward[3], step)
        self.log_expert.scalar_summary('max reward', self.expert_reward[3], step)


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)