import os
import time as timer
from data_gen.utils import *
from settings import FRAME_SIZE, DATA_DIR
import pickle
from mj_envs.mujoco_env import MujocoEnv

class MujocoEnv(MujocoEnv):
    """Superclass for all MuJoCo environments .
    """
    def __init__(self, model_path, frame_skip):
        self.use_tactile = True
        super().__init__(model_path, frame_skip)

    def get_body_com(self, body_name):
        return self.sim.data.get_body_xpos(body_name)

    def visualize_policy(self, policy, horizon=1000, num_episodes=1, mode='exploration', use_img=False, use_seq=False,
                         has_robot_info=True, frame_size=FRAME_SIZE, camera_name=None):
        self.mujoco_render_frames = True
        for ep in range(num_episodes):

            o, prev_o, prev_prev_o, t, d, robot_info = self.init(has_robot_info, use_img, camera_name, frame_size)
            while t < horizon and d is False:
                o, r, d, env_info, prev_o, prev_prev_o, robot_info = self.single_step(policy,
                                                                                      o, prev_o, prev_prev_o,
                                                                                      use_img, use_seq,
                                                                                      robot_info, has_robot_info,
                                                                                      mode)
                t = t + 1

        self.mujoco_render_frames = False

    def get_pixels(self, frame_size=FRAME_SIZE, camera_name=None, device_id=0):
        pixels = self.sim.render(width=frame_size[0], height=frame_size[1],
                                 mode='offscreen', camera_name=camera_name, device_id=device_id)

        pixels = pixels[::-1, :, :]
        return pixels

    def visualize_policy_offscreen(self, policy, horizon=1000,
                                   num_episodes=1,
                                   frame_size=(640, 480),
                                   mode='exploration',
                                   save_loc='/tmp/',
                                   filename='newvid',
                                   camera_name=None,
                                   use_img=False,
                                   use_seq=False,
                                   pickle_dump=True,
                                   has_robot_info=True):
        import skvideo.io
        for ep in range(num_episodes):
            print("Episode %d: rendering offline " % ep, end='', flush=True)
            arrs = []
            t0 = timer.time()
            o, prev_o, prev_prev_o, t, d, robot_info = self.init(has_robot_info, use_img, camera_name, frame_size)
            while t < horizon and d is False:
                o, r, d, env_info, prev_o, prev_prev_o, robot_info = self.single_step(policy,
                                                                                      o, prev_o, prev_prev_o,
                                                                                      use_img, use_seq,
                                                                                      robot_info, has_robot_info,
                                                                                      mode)
                t = t + 1
                curr_frame = self.sim.render(width=frame_size[0], height=frame_size[1],
                                             mode='offscreen', camera_name=camera_name, device_id=0)
                arrs.append(curr_frame[::-1, :, :])
                print(t, end=', ', flush=True)
            file_name = save_loc + filename + str(ep) + ".mp4"
            if not pickle_dump:
                skvideo.io.vwrite(file_name, np.asarray(arrs))
            else:
                file_name += '.pickle'
                pickle.dump(np.asarray(arrs), open(file_name, 'wb'))
            print("saved", file_name)
            t1 = timer.time()
            print("time taken = %f" % (t1 - t0))

    def init(self, has_robot_info, use_img, camera_name=None, frame_size=FRAME_SIZE):
        o = self.reset()
        robot_info = None
        if has_robot_info:
            o, env_info = self.reset()
            robot_info = env_info['robot_info']

        if use_img:
            o = self.get_pixels(camera_name=camera_name)

        d = False
        t = 0
        prev_o = o
        prev_prev_o = o

        return o, prev_o, prev_prev_o, t, d, robot_info

    def single_step(self, policy, o, prev_o, prev_prev_o, use_img, use_seq, robot_info, has_robot_info, mode):
        o_c = o
        prev_o_c = prev_o
        if use_seq:
            prev_prev_o = np.expand_dims(prev_prev_o, axis=0)
            prev_o = np.expand_dims(prev_o, axis=0)
            o = np.expand_dims(o, axis=0)

            o = np.concatenate((prev_prev_o, prev_o, o), axis=0)
            a = policy.get_action(o, use_seq=use_seq, robot_info=robot_info)[0] if mode == 'exploration' else \
                policy.get_action(o, use_seq=use_seq, robot_info=robot_info)[1]['evaluation']
            # Maybe clip here too?
        else:
            a = policy.get_action(o)[0] if mode == 'exploration' else \
                policy.get_action(o)[1]['evaluation']
            # Not so sure about this
            a = np.clip(a, -1.0, 1.0)

        prev_prev_o = prev_o_c
        prev_o = o_c
        o, r, d, env_info = self.step(a)

        if has_robot_info:
            robot_info = env_info['robot_info']

        if use_img:
            o = self.get_pixels(frame_size=FRAME_SIZE)

        return o, r, d, env_info, prev_o, prev_prev_o, robot_info


    def visualize_policy_data(self, data):
        import skvideo.io
        data_points = [10, 20, 30, 50]
        for i in data_points:
            file_name = os.path.join(DATA_DIR, 'data_videos', 'data_point_%d.mp4' % i)
            skvideo.io.vwrite(file_name, (data[i]["image_pixels"] * 255).astype(np.int32))
            print("saved", file_name)

    def visualize_policy_data_single(self, data):
        import skvideo.io
        file_name = os.path.join(DATA_DIR, 'data_videos', 'data_point_%d.mp4' % np.random.randint(10000))
        skvideo.io.vwrite(file_name, data)
        print("saved", file_name)
