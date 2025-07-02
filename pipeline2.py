#!/usr/bin/env python

# Standard Library Imports
import math
import random
import rospy
import os
import time
import json
import re
import pickle
import networkx as nx
import matplotlib.pyplot as plt

# ROS and ROS messages
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from actionlib import SimpleActionClient
from pal_interaction_msgs.msg import TtsAction, TtsGoal

# OpenAI and Audio Processing Libraries
from openai import OpenAI
import librosa
import soundfile as sf

# Computer Vision and Point Cloud Processing
import cv2
import open3d as o3d
import numpy as np
import base64
from matplotlib.colors import to_rgb
import tf2_ros

# Custom imports
from utils import *
from scene_acquisition import *
from agents import *
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

# Initialize OpenAI client
client = OpenAI()
rospy.init_node('empower_pipeline_final', anonymous=True)

# Task definitions
tasks = {
    "T1_task_1": "use the fire extinguisher",
    "T1_task_2": "lift the table",
    "T1_task_3": "detach the TV from the wall",
    "T1_task_4": "Exit climbing the stairs",
    "T2_task_1": "move the juice fruit to the left of the glass",
    "T2_task_2": "Fry an egg",
    "T2_task_3": "Give me the screwdriver",
    "T3_task_1": "Unplug the electrical cable",
    "T3_task_2": "Put screwdriver into the electric plug",
    "T3_task_3": "Pour some water on the pc",
}


class PipelineFinal():
    def __init__(self, task_id="task_1"):
        self.loader_instance = None
        self.COLORS = ['red', 'green', 'blue', 'magenta', 'cyan', 'yellow'] * 3
        self.task_id = task_id
        self.task = tasks.get(task_id, "Pick the TV and laptop")
        
        # Set up logging directories
        self.setup_logging()
        
        # ROS components
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.bridge = CvBridge()
        
        # Publishers
        self.publisher_centroid = rospy.Publisher("/pcl_centroids", MarkerArray, queue_size=100)
        self.publisher_maximum = rospy.Publisher("/pcl_maximum", MarkerArray, queue_size=100)
        self.publisher_names = rospy.Publisher("/pcl_names", MarkerArray, queue_size=100)
        
        # Initialize prompts paths
        self.visual_prompt_path = PROMPT_DIR + "visual_agent_prompt.txt"
        self.conversational_prompt_path = PROMPT_DIR + "conversational_prompt.txt"
        self.planner_prompt_path = PROMPT_DIR + "planner_prompt.txt"
    def setup_logging(self):
        """Setup logging directories and files"""
        self.save_folder = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "experiments",
            self.task_id, 
            time.strftime("%Y%m%d-%H%M%S")
        )
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder, exist_ok=True)
            
        self.point_cloud_path = os.path.join(self.save_folder, "point_cloud.pcd")
        self.img_path = os.path.join(self.save_folder, "image.png")
        self.text_path = os.path.join(self.save_folder, "logs.txt")
        self.time_log_path = os.path.join(self.save_folder, "time_logs.txt")
        self.audio_path = os.path.join(self.save_folder, "whisper_audio.wav")
        self.log_file = open(self.text_path, "w")
        self.time_log_file = open(self.time_log_path, "w")
        
        # Initialize timing variables
        self.phase_start_time = None
        self.total_start_time = time.time()
        
        # Log initial setup
        self.log_file.write("=" * 50 + "\n")
        self.log_file.write("EMPOWER PIPELINE EXECUTION LOG\n")
        self.log_file.write("=" * 50 + "\n")
        self.log_file.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_file.write(f"Task ID: {self.task_id}\n")
        self.log_file.write(f"Task Description: {self.task}\n")
        self.log_file.write(f"Save Folder: {self.save_folder}\n")
        self.log_file.write("=" * 50 + "\n\n")
        self.log_file.flush()
        
        # Initialize time log file
        self.time_log_file.write("=" * 60 + "\n")
        self.time_log_file.write("EMPOWER PIPELINE EXECUTION TIME LOG\n")
        self.time_log_file.write("=" * 60 + "\n")
        self.time_log_file.write(f"Pipeline Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.time_log_file.write(f"Task ID: {self.task_id}\n")
        self.time_log_file.write(f"Task Description: {self.task}\n")
        self.time_log_file.write("=" * 60 + "\n\n")
        self.time_log_file.flush()

    def start_phase_timer(self, phase_name):
        """Start timing a phase"""
        self.phase_start_time = time.time()
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        self.time_log_file.write(f"[{timestamp}] STARTED: {phase_name}\n")
        self.time_log_file.flush()
        
    def end_phase_timer(self, phase_name):
        """End timing a phase and log duration"""
        if self.phase_start_time is not None:
            duration = time.time() - self.phase_start_time
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            self.time_log_file.write(f"[{timestamp}] COMPLETED: {phase_name}\n")
            self.time_log_file.write(f"  Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)\n\n")
            self.time_log_file.flush()
            self.phase_start_time = None
            return duration
        return 0

    def rgb_to_bgr(self, rgb_color):
        r, g, b = rgb_color
        return [b, g, r]

    def set_loader(self, loader_instance):
        self.loader_instance = loader_instance



    def split_word(self, words):
        splitted_word = []
        words = words.lower()
        doc = self.loader_instance.nlp(words)
        for token in doc:
            if token.pos_ == "AUX" or (token.pos_ == "NOUN" and token.dep_ in ["dobj", "ROOT", "nsubj"]) or token.pos_ == "VERB" or token.pos_ == "PROPN":
                splitted_word.append(token.text)
        return splitted_word

    def compare_two_list_of_objects(self, position_in_image_first, position_in_image_second, relation, object_first, object_second):
        if position_in_image_first == {} or position_in_image_second == {}:
            return

        if "on" in relation:
            min_distance_x = 30000
            for key_first in position_in_image_first.keys():
                for key_second in position_in_image_second.keys():
                    dis_x = abs(position_in_image_first[key_first]['x'] - position_in_image_second[key_second]['x'])
                    dis_y = abs(position_in_image_first[key_first]['y'] - position_in_image_second[key_second]['y'])
                    distance = math.sqrt((dis_x * dis_x + dis_y * dis_y))
                    if min_distance_x > dis_y:
                        index_first = key_first
                        index_second = key_second
                        min_distance_x = distance

            self.data_reordered[index_first]['label'] = object_first
            self.data_reordered[index_second]['label'] = object_second

        if "left" in relation:
            min_distance_x = 30000
            min_distance_y = 10000
            for key_first in position_in_image_first.keys():
                for key_second in position_in_image_second.keys():
                    min_distance_x_bb = abs(position_in_image_first[key_first]['x'] - position_in_image_second[key_second]['x']) // 2
                    min_distance_y_bb = abs(position_in_image_first[key_first]['y'] - position_in_image_second[key_second]['y']) // 2
                    if min_distance_x_bb < min_distance_x and min_distance_x_bb != 0 and min_distance_y > min_distance_y_bb:
                        index_first = key_first
                        index_second = key_second
                        min_distance_x = min_distance_x_bb
                        min_distance_y = min_distance_y_bb

            self.data_reordered[index_first]['label'] = object_first
            self.data_reordered[index_second]['label'] = object_second
            self.dict_detections.pop(index_first)
            self.dict_detections.pop(index_second)

        if "right" in relation:
            min_distance_x = 30000
            min_distance_y = 10000
            for key_first in position_in_image_first.keys():
                for key_second in position_in_image_second.keys():
                    min_distance_x_bb = abs(position_in_image_first[key_first]['x'] - position_in_image_second[key_second]['x']) // 2
                    min_distance_y_bb = abs(position_in_image_first[key_first]['y'] - position_in_image_second[key_second]['y']) // 2
                    if min_distance_x_bb < min_distance_x and min_distance_x_bb != 0 and min_distance_y > min_distance_y_bb and min_distance_y_bb != 0:
                        index_first = key_first
                        index_second = key_second
                        min_distance_x = min_distance_x_bb
                        min_distance_y = min_distance_y_bb
            self.data_reordered[index_first]['label'] = object_first
            self.data_reordered[index_second]['label'] = object_second
            self.dict_detections.pop(index_first)
            self.dict_detections.pop(index_second)

    def obtain_bb_grounded(self, index_first, index_second, relation, object_first, object_second):
        detection_data = self.dict_detections
        position_in_image_first = {}
        position_in_image_second = {}
        for i in range(len(index_first)):
            if index_first[i] not in position_in_image_first.keys():
                position_in_image_first[index_first[i]] = {'x': None, 'y': None}
            position_in_image_first[index_first[i]]['x'] = (detection_data[index_first[i]]['bbox'][0] + detection_data[index_first[i]]['bbox'][2]) // 2
            position_in_image_first[index_first[i]]['y'] = (detection_data[index_first[i]]['bbox'][1] + detection_data[index_first[i]]['bbox'][3]) // 2
        for i in range(len(index_second)):
            if index_second[i] not in position_in_image_second.keys():
                position_in_image_second[index_second[i]] = {'x': None, 'y': None}
            position_in_image_second[index_second[i]]['x'] = (detection_data[index_second[i]]['bbox'][0] + detection_data[index_second[i]]['bbox'][2]) // 2
            position_in_image_second[index_second[i]]['y'] = (detection_data[index_second[i]]['bbox'][1] + detection_data[index_second[i]]['bbox'][3]) // 2
        self.compare_two_list_of_objects(position_in_image_first, position_in_image_second, relation, object_first, object_second)

    def show_mask(self, mask, random_color=True):
        if random_color:
            color = np.concatenate([np.random.random(3)], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        mask_image = mask_image.numpy() * 255
        mask_image = mask_image.astype(np.uint8)
        return mask_image

    def find_bb_relation(self, relation_object):
        index_ = []
        for index, detection in self.dict_detections.items():
            if detection['label'].lower() in relation_object.lower():
                index_.append(index)
        return index_

    def get_R_and_T(self, trans):
        Tx_base = trans.transform.translation.x
        Ty_base = trans.transform.translation.y
        Tz_base = trans.transform.translation.z
        T = np.array([Tx_base, Ty_base, Tz_base])
        # Quaternion coordinates
        qx = trans.transform.rotation.x
        qy = trans.transform.rotation.y
        qz = trans.transform.rotation.z
        qw = trans.transform.rotation.w

        # Rotation matrix
        R = 2 * np.array([[pow(qw, 2) + pow(qx, 2) - 0.5, qx * qy - qw * qz, qw * qy + qx * qz], [qw * qz + qx * qy, pow(qw, 2) + pow(qy, 2) - 0.5, qy * qz - qw * qx], [qx * qz - qw * qy, qw * qx + qy * qz, pow(qw, 2) + pow(qz, 2) - 0.5]])
        return R, T

    def set_marker(self, point, color, id, R, T):
        point = point / 1000
        transform = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        point = np.dot(transform, point)  # in xtion
        R = np.transpose(R)
        point = np.dot(R, point - T)  # in map

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time(0)
        marker.pose.position.x = point[0]
        marker.pose.position.y = point[1]
        marker.pose.position.z = point[2]
        marker.pose.orientation.x = 0
        marker.pose.orientation.y = 0
        marker.pose.orientation.z = 0
        marker.pose.orientation.w = 1
        marker.type = marker.SPHERE
        color = to_rgb(color)
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 1.0
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.id = id

        return marker

    def set_names(self, marker, text):
        """
        Creates a text marker to display object names above the centroid
        Args:
            marker: The centroid marker to position the text above
            text: The text to display (object label)
        Returns:
            Marker: Text marker to display in RViz
        """
        text_marker = Marker()
        text_marker.header.frame_id = marker.header.frame_id
        text_marker.header.stamp = marker.header.stamp
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD

        # Position text slightly above the centroid
        text_marker.pose.position.x = marker.pose.position.x
        text_marker.pose.position.y = marker.pose.position.y
        text_marker.pose.position.z = marker.pose.position.z + 0.1  # 10cm above the centroid

        text_marker.pose.orientation.x = 0.0
        text_marker.pose.orientation.y = 0.0
        text_marker.pose.orientation.z = 0.0
        text_marker.pose.orientation.w = 1.0

        text_marker.text = text
        text_marker.scale.z = 0.15  # Text size
        text_marker.color = marker.color  # Same color as the centroid
        text_marker.id = marker.id + 1000  # Different ID to avoid conflicts

        return text_marker

    # New methods from new_pipeline.py
    def depth_image_to_point_cloud(self, depth_image, camera_intrinsics):
        height, width = depth_image.shape
        points = []

        v, u = np.indices((height, width))

        x = (u - camera_intrinsics[0, 2]) * depth_image / camera_intrinsics[0, 0]
        y = (v - camera_intrinsics[1, 2]) * depth_image / camera_intrinsics[1, 1]
        z = depth_image

        points = np.dstack((x, y, z)).reshape(-1, 3)

        return points

    def capture_pcd(self):
        msg_img_g = rospy.wait_for_message("/xtion/depth/image_raw", Image)
        camera_info = rospy.wait_for_message("/xtion/depth/camera_info", CameraInfo)
        proj_matrix = camera_info.K
        fx = proj_matrix[0]
        fy = proj_matrix[4]
        cx = proj_matrix[2]
        cy = proj_matrix[5]

        camera_intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        img_g = self.bridge.imgmsg_to_cv2(msg_img_g)
        depth_image = np.asarray(img_g)
        point_cloud = self.depth_image_to_point_cloud(depth_image, camera_intrinsics)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)

        pcd.transform(np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]))
        o3d.io.write_point_cloud(self.point_cloud_path, pcd)
        return pcd

    def create_graph(self, json_structure, save_path):
        G = nx.Graph()
        for obj in json_structure['Objects']:
            label = obj['Label']
            relations = obj['Relations']
            G.add_node(label)

            for relation in relations:
                relation_attribute, label2 = relation.split(";")[0], relation.split(";")[1]
                G.add_edge(label, label2, relation=relation_attribute)

        pos = nx.spring_layout(G, k=0.5, iterations=50)  # Increase k for more space between nodes

        nx.draw(G, pos, with_labels=True, node_size=300, node_color="skyblue", font_size=5, font_weight='bold', width=2)

        edge_labels = nx.get_edge_attributes(G, 'relation')

        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=5, font_color='blue')

        # Save the graph
        plt.savefig(save_path)
        plt.close()

    def say_phrase(self, data):
        client_tts = SimpleActionClient('/tts', TtsAction)
        client_tts.wait_for_server()
        goal = TtsGoal()
        goal.rawtext.text = data
        goal.rawtext.lang_id = "en_GB"
        client_tts.send_goal_and_wait(goal)

    def listen_for(self, seconds):
        try:
            seconds = seconds // 3
            rospy.loginfo(f"Received audio message")
            whisper_audio = []
            for _ in range(seconds):
                data = rospy.wait_for_message("/data_topic", Float32MultiArray)
                audio_array = np.array(data.data)
                amplitude_audio = np.abs(audio_array)
                amplitude_audio = np.mean(amplitude_audio)
                rospy.loginfo(f"Amplitude: {amplitude_audio}")
                if np.max(amplitude_audio) < 0.001:
                    rospy.loginfo("Audio troppo silenzioso, ignorato.")
                else:
                    audio = librosa.resample(audio_array, orig_sr=44100, target_sr=16000).astype(np.float32)
                    whisper_audio = np.concatenate((whisper_audio, np.array(audio, dtype=np.float32)))

            sf.write(self.audio_path, whisper_audio, 16000)

            whisper_audio = open(self.audio_path, "rb")
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=whisper_audio,
                language='en')
            os.system("rm " + self.audio_path)
            return transcription.text
        except Exception as e:
            rospy.logerr(f"Error in callback: {e}")

    def scene_description_with_llm(self, visual_prompt ,img):
        self.start_phase_timer("Scene Description with LLM")
        
        self.log_file.write("SCENE DESCRIPTION PHASE\n")
        self.log_file.write("-" * 30 + "\n")
        self.log_file.write(f"Visual Prompt: {visual_prompt[:200]}...\n")
        self.log_file.write(f"Task for scene analysis: {self.task}\n\n")
        
        # Time VLM call
        vlm_start = time.time()
        json_answer = vlm_call(visual_prompt + " objects for this task: " + self.task, img)
        vlm_duration = time.time() - vlm_start
        self.time_log_file.write(f"  VLM Call Duration: {vlm_duration:.2f} seconds\n")
        
        scene = json_answer.replace("```json", "").replace("```","")
        
        self.log_file.write("VLM Response (Raw):\n")
        self.log_file.write(json_answer + "\n\n")
        
        try:
            json_structure = json.loads(scene)
            self.log_file.write("Scene Structure (Parsed JSON):\n")
            self.log_file.write(json.dumps(json_structure, indent=2) + "\n\n")
        except json.JSONDecodeError as e:
            self.log_file.write(f"JSON Parsing Error: {e}\n")
            self.log_file.write(f"Raw scene text: {scene}\n\n")
            self.end_phase_timer("Scene Description with LLM")
            return None
            
        json_file_path = os.path.join(self.save_folder, time.strftime("%Y%m%d-%H%M%S") + ".json")
        graph_file_path = os.path.join(self.save_folder, time.strftime("%Y%m%d-%H%M%S") + ".png")
        
        # Time file operations
        file_ops_start = time.time()
        with open(json_file_path, 'w') as outfile:
            json.dump(json_structure, outfile, indent=4)
            
        self.create_graph(json_structure, graph_file_path)
        file_ops_duration = time.time() - file_ops_start
        self.time_log_file.write(f"  File Operations Duration: {file_ops_duration:.2f} seconds\n")
        
        self.log_file.write(f"Scene JSON saved to: {json_file_path}\n")
        self.log_file.write(f"Scene graph saved to: {graph_file_path}\n")
        self.log_file.write("-" * 30 + "\n\n")
        self.log_file.flush()
        
        self.end_phase_timer("Scene Description with LLM")
        return json_structure

    def extract_first_question(self, text):
        question_pattern = r"<QUESTION>(.*?)</QUESTION>"
        match = re.search(question_pattern, text, re.DOTALL)
        if match:
            # Extract and return the first question found
            return match.group(1).strip()
        return None

    def conversational_planning(self, encoded_image, json_scene):
        """Enhanced conversational planning process"""
        self.start_phase_timer("Conversational Planning")
        
        self.log_file.write("CONVERSATIONAL PLANNING PHASE\n")
        self.log_file.write("-" * 35 + "\n")
        
        # Load prompts
        prompt_load_start = time.time()
        conversational_prompt = open(self.conversational_prompt_path, "r").read()
        conversational_prompt = conversational_prompt.replace("<SCENE_DESCRIPTION>", json.dumps(json_scene))
        conversational_prompt = conversational_prompt.replace("<TASK>", self.task)
        
        planner_prompt = open(self.planner_prompt_path, "r").read()
        planner_prompt = planner_prompt.replace("<SCENE_DESCRIPTION>", json.dumps(json_scene))
        prompt_load_duration = time.time() - prompt_load_start
        self.time_log_file.write(f"  Prompt Loading Duration: {prompt_load_duration:.2f} seconds\n")
        
        self.log_file.write("Conversational Prompt Template Loaded\n")
        self.log_file.write("Planner Prompt Template Loaded\n\n")
        
        conversation = "<CONVERSATION>\n HUMAN: " + self.task
        self.log_file.write("Initial Conversation:\n")
        self.log_file.write(conversation + "\n\n")
        
        # Time initial LLM call
        initial_llm_start = time.time()
        llm_answer = llm_call(conversational_prompt, conversation)
        initial_llm_duration = time.time() - initial_llm_start
        self.time_log_file.write(f"  Initial LLM Call Duration: {initial_llm_duration:.2f} seconds\n")
        
        self.log_file.write("Initial LLM Response:\n")
        self.log_file.write(llm_answer + "\n\n")
        
        print("Initial LLM response:", llm_answer)
        
        # Conversational loop for clarification
        question_count = 0
        conversation_loop_start = time.time()
        while "<PLAN>" not in llm_answer:
            question = self.extract_first_question(llm_answer)
            
            if question is None:
                self.log_file.write("No more questions found in LLM response. Ending conversation loop.\n\n")
                break
            else:
                question_count += 1
                self.log_file.write(f"Question #{question_count}: {question}\n")
                
                # Time TTS
                tts_start = time.time()
                self.say_phrase(question)
                tts_duration = time.time() - tts_start
                self.time_log_file.write(f"  TTS #{question_count} Duration: {tts_duration:.2f} seconds\n")
                
                print("Question:", question)
                
                # Time user input
                input_start = time.time()
                answer = input("Answer: ")
                if answer == "EXIT":
                    llm_answer = "<PLAN>N\A</PLAN>"
                    break
                input_duration = time.time() - input_start
                self.time_log_file.write(f"  User Input #{question_count} Duration: {input_duration:.2f} seconds\n")
                
                self.log_file.write(f"Human Answer #{question_count}: {answer}\n\n")
                
                conversation = conversation + "\nYOU: " + question + "\n HUMAN new message: " + answer
                self.log_file.write("Updated Conversation:\n")
                self.log_file.write(conversation + "\n\n")
                
                # Time follow-up LLM call
                followup_llm_start = time.time()
                llm_answer = llm_call(conversational_prompt, conversation)
                followup_llm_duration = time.time() - followup_llm_start
                self.time_log_file.write(f"  Follow-up LLM Call #{question_count} Duration: {followup_llm_duration:.2f} seconds\n")
                
                self.log_file.write(f"LLM Response #{question_count + 1}:\n")
                self.log_file.write(llm_answer + "\n\n")
                print("LLM response:", llm_answer)
        
        conversation_loop_duration = time.time() - conversation_loop_start
        self.time_log_file.write(f"  Total Conversation Loop Duration: {conversation_loop_duration:.2f} seconds\n")
        self.time_log_file.write(f"  Number of Questions Asked: {question_count}\n")
        
        self.log_file.write("Conversation phase completed. Generating final plan...\n\n")
        print("No more questions to ask. Generating final plan...")
        
        # Time final plan generation
        final_plan_start = time.time()
        llm_plan = llm_call(planner_prompt, llm_answer)
        final_plan_duration = time.time() - final_plan_start
        self.time_log_file.write(f"  Final Plan Generation Duration: {final_plan_duration:.2f} seconds\n")
        
        self.log_file.write("FINAL PLAN (With Awareness):\n")
        self.log_file.write(llm_plan + "\n\n")
        print("Final plan:", llm_plan)
        
        # Time plan without awareness
        no_awareness_start = time.time()
        vlm_answer, description_agent_answer, planner_agent_answer  = multi_agent_planning(self.img_path, self.task)
        no_awareness_duration = time.time() - no_awareness_start
        self.time_log_file.write(f"  Plan Without Awareness Duration: {no_awareness_duration:.2f} seconds\n")
        
        self.log_file.write("PLAN WITHOUT AWARENESS (Comparison):\n")
        self.log_file.write(planner_agent_answer + "\n")
        self.log_file.write("-" * 35 + "\n\n")
        self.log_file.flush()
        
        print("Plan without awareness:", planner_agent_answer)
        
        self.end_phase_timer("Conversational Planning")
        return llm_plan, planner_agent_answer

    def color_pcl(self, image, pcd, detections):
        pcd.colors = o3d.utility.Vector3dVector(np.tile(to_rgb('gray'), (len(pcd.points), 1)))

        h, w, _ = image.shape

        masks = []
        masks_flipped = []

        for key in detections.keys():
            mask = detections[key]['mask']
            masks.append(mask[:, :, 0])
            masks_flipped.append(cv2.flip(mask[:, :, 0], 1))
        camera_info = rospy.wait_for_message("/xtion/depth/camera_info", CameraInfo)
        proj_matrix = camera_info.K
        fx = proj_matrix[0]
        fy = proj_matrix[4]
        cx = proj_matrix[2]
        cy = proj_matrix[5]

        colors_dict = {}

        for idx, point in enumerate(pcd.points):
            # Remove the depth check that was filtering all points
            x_ = point[0]
            y_ = point[1]
            z_ = point[2]

            # Avoid division by zero
            if abs(z_) < 0.001:
                continue

            x = int((fx * x_ / z_) + cx)
            y = int((fy * y_ / z_) + cy)

            for id_color, (mask, mask_flipped) in enumerate(zip(masks, masks_flipped)):
                if 0 <= x < w and 0 <= y < h and mask[y, x] != 0:
                    image[y, x] = self.rgb_to_bgr([int(color * 255) for color in to_rgb(self.COLORS[id_color])])

                if 0 <= x < w and 0 <= y < h and mask_flipped[y, x] != 0:
                    if id_color not in colors_dict.keys():
                        colors_dict[id_color] = []
                    pcd.colors[idx] = to_rgb(self.COLORS[id_color])
                    colors_dict[id_color].append(point)

        array_centroids = MarkerArray()
        array_maximum = MarkerArray()
        array_name = MarkerArray()
        trans = self.tf_buffer.lookup_transform("xtion_rgb_optical_frame", "map", rospy.Time(0), rospy.Duration(2.0))
        Rx2m, Tx2m = self.get_R_and_T(trans)

        for id_color, list_points in colors_dict.items():
            centroid = np.mean(list_points, axis=0)
            new_list = []
            for point in list_points:
                if np.linalg.norm(point - centroid) < 100:
                    new_list.append(point)

            if new_list == []:
                new_centroid = centroid
                new_list = list_points
            else:
                new_centroid = np.mean(new_list, axis=0)
            label = detections[id_color]['label'] if id_color in detections else f"Object {id_color}"

            # Create unique marker IDs using time prefix
            time_id_prefix = random.randint(1, 1000)
            rospy.loginfo(f"Using marker time ID prefix: {time_id_prefix}")

            new_marker = self.set_marker(new_centroid, self.COLORS[id_color], time_id_prefix, Rx2m, Tx2m)
            array_centroids.markers.append(new_marker)
            max_point = np.max(new_list, axis=0)
            # Use the same unique ID with an offset for the maximum marker
            array_maximum.markers.append(self.set_marker(max_point, self.COLORS[id_color], time_id_prefix + 200, Rx2m, Tx2m))
            # Use the same unique ID with a different offset for the name marker
            name_marker = self.set_names(new_marker, label)
            name_marker.id = time_id_prefix + 300  # Override the ID set in set_names
            array_name.markers.append(name_marker)
            print("Marker processed")

        self.publisher_centroid.publish(array_centroids)
        self.publisher_maximum.publish(array_maximum)
        self.publisher_names.publish(array_name)
        print("Markers published")
        cv2.imwrite(os.path.join(self.save_folder, 'colored_image.png'), image)
        #o3d.visualization.draw_geometries([pcd])
        return

    def run_pipeline(self):
        """Main pipeline execution with enhanced conversational planning"""
        self.start_phase_timer("Complete Pipeline Execution")
        
        self.log_file.write("PIPELINE EXECUTION STARTED\n")
        self.log_file.write("=" * 50 + "\n\n")
        
        # Acquire image
        self.start_phase_timer("Image Acquisition")
        self.log_file.write("IMAGE ACQUISITION PHASE\n")
        self.log_file.write("-" * 25 + "\n")
        
        image, depth = acquire_image()
        self.log_file.write("Image and depth data acquired successfully\n")
        print("Image acquired")
        
        # Save RGB image for processing
        cv2.imwrite(self.img_path, image)
        self.log_file.write(f"RGB image saved to: {self.img_path}\n")
        self.end_phase_timer("Image Acquisition")
        
        # Capture point cloud
        self.start_phase_timer("Point Cloud Capture")
        pcd = self.capture_pcd()
        self.log_file.write(f"Point cloud captured and saved to: {self.point_cloud_path}\n")
        self.log_file.write(f"Point cloud contains {len(pcd.points)} points\n")
        self.log_file.write("-" * 25 + "\n\n")
        print("Point cloud captured")
        self.end_phase_timer("Point Cloud Capture")
        
        # Encode image for LLM processing
        self.start_phase_timer("Image Encoding")
        with open(self.img_path, "rb") as im_file:
            encoded_image = base64.b64encode(im_file.read()).decode("utf-8")
        
        self.log_file.write(f"Image encoded for LLM processing (size: {len(encoded_image)} characters)\n\n")
        print(f"Task: {self.task}")
        self.end_phase_timer("Image Encoding")
        
        # Enhanced scene understanding with conversational AI
        visual_prompt = open(self.visual_prompt_path, "r").read()
        json_scene = self.scene_description_with_llm(visual_prompt, encoded_image)
        
        if json_scene is None:
            self.log_file.write("ERROR: Scene description failed. Stopping pipeline.\n")
            self.time_log_file.write("ERROR: Pipeline stopped due to scene description failure.\n")
            self.log_file.close()
            self.time_log_file.close()
            return
            
        print(json_scene)
        
        # Conversational planning process
        final_plan, plan_no_awareness = self.conversational_planning(encoded_image, json_scene)
        
        # Save results
        self.start_phase_timer("Results Saving")
        self.log_file.write("RESULTS SAVING PHASE\n")
        self.log_file.write("-" * 20 + "\n")
        
        results_enhanced = {
            "task": self.task,
            "scene_description": json_scene,
            "final_plan": final_plan,
            "plan_no_awareness": plan_no_awareness,
        }
        
        pickle_path = os.path.join(self.save_folder, "enhanced_planning.pkl")
        txt_path = os.path.join(self.save_folder, "enhanced_planning.txt")
        
        with open(pickle_path, 'wb') as f:
            pickle.dump(results_enhanced, f, protocol=2)
        
        with open(txt_path, 'w') as f:
            f.write("Task: ")
            f.write(results_enhanced["task"])
            f.write("\nScene Description: ")
            f.write(str(results_enhanced["scene_description"]))
            f.write("\nFinal Plan: ")
            f.write(results_enhanced["final_plan"])
            f.write("\nPlan without awareness: ")
            f.write(results_enhanced["plan_no_awareness"])
        
        self.log_file.write(f"Enhanced results saved to: {pickle_path}\n")
        self.log_file.write(f"Enhanced results saved to: {txt_path}\n")
        self.log_file.write("-" * 20 + "\n\n")
        
        print("Enhanced results saved to file.")
        self.end_phase_timer("Results Saving")
        
        # Continue with traditional object detection if loader is available
        if self.loader_instance is not None:
            self.log_file.write("TRADITIONAL OBJECT DETECTION PHASE\n")
            self.log_file.write("-" * 35 + "\n")
            self.run_traditional_detection(image, depth, json_scene)
        else:
            self.log_file.write("SKIPPING TRADITIONAL DETECTION\n")
            self.log_file.write("-" * 30 + "\n")
            self.log_file.write("Loader instance not set. Skipping traditional object detection.\n\n")
            rospy.logwarn("Loader instance not set. Skipping traditional object detection.")
        
        # Log total execution time
        total_duration = self.end_phase_timer("Complete Pipeline Execution")
        
        self.time_log_file.write("=" * 60 + "\n")
        self.time_log_file.write("PIPELINE EXECUTION SUMMARY\n")
        self.time_log_file.write("=" * 60 + "\n")
        self.time_log_file.write(f"Total Pipeline Duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)\n")
        self.time_log_file.write(f"Pipeline End Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.time_log_file.write("=" * 60 + "\n")
        
        self.log_file.write("PIPELINE EXECUTION COMPLETED\n")
        self.log_file.write("=" * 50 + "\n")
        self.log_file.close()
        self.time_log_file.close()


    def compare_two_words(self, list1, list2):
        if isinstance(list1, str):
            list1 = list1.split() 
        if isinstance(list2, str):
            list2 = list2.split() 
        word1 = list1.copy()
        word2 = list2.copy()
        min_1 = len(word1)
        min_2 = len(word2)
        sim_word = []
        if min_1 > min_2:
            for index in range(min_1):
                found = False
                sim = 0
                min_2 = len(word2)
                for index_2 in range(min_2):
                    sim = (self.loader_instance.wv.similarity(word1[index], word2[index_2]))
                    if sim > 0.708:
                        found = True
                        sim_word.append(sim)
                        word2.pop(index_2)
                        break
                if not found:
                    sim_word.append(sim)
        else:
            for index in range(min_2):
                found = False
                sim = 0
                min_1 = len(word1)
                for index_2 in range(min_1):
                    sim = (self.loader_instance.wv.similarity(word2[index], word1[index_2]))
                    if sim > 0.708:
                        found = True
                        sim_word.append(sim)
                        word1.pop(index_2)
                        break
                if not found:
                    sim_word.append(sim)
        if sim_word == []:
            return None
        sim_word = np.mean(sim_word)
        return sim_word

    def is_in_list(self, word, list):
        for obj in list:
            if self.compare_two_words(word, obj) != None and self.compare_two_words(word, obj) > 0.708:
                return True
        return False


    def get_classes(self, object_relations):
        relation_list = []

        # Iterate through each object in the list
        for obj in object_relations:
            if not self.is_in_list(obj["Label"], relation_list):
                relation_list.append(obj["Label"])

        return relation_list

    
    def process_relations(self, object_relations):
        try:
            self.log_file.write("Processing Object Relations:\n")
            relation_count = 0
            
            for obj in object_relations:
                # Extract the label (or name) of the object
                object_label = obj["Label"]
                
                # Extract the relations associated with this object
                for relation in obj["Relations"]:
                    relation_count += 1
                    self.log_file.write(f"  Relation {relation_count}: {relation}\n")
                    
                    # The relations are in the format "on; floor", "behind; desk", etc.
                    relation_parts = relation.split(";")
                    
                    if len(relation_parts) == 2:
                        relation_object_first = object_label.strip()  # First object (e.g., "desk")
                        relation_type = relation_parts[0].strip()  # Relation type (e.g., "on")
                        relation_object_second = relation_parts[1].strip()  # Second object (e.g., "floor")
                        
                        self.log_file.write(f"    Object 1: {relation_object_first}\n")
                        self.log_file.write(f"    Relation: {relation_type}\n")
                        self.log_file.write(f"    Object 2: {relation_object_second}\n")
                        
                        # Find bounding box indices for both objects
                        index_bounding_box_first = self.find_bb_relation(relation_object_first)
                        index_bounding_box_second = self.find_bb_relation(relation_object_second)
                        
                        self.log_file.write(f"    BB indices for {relation_object_first}: {index_bounding_box_first}\n")
                        self.log_file.write(f"    BB indices for {relation_object_second}: {index_bounding_box_second}\n")

                        # If either bounding box index is found, process further
                        if index_bounding_box_first or index_bounding_box_second:
                            self.obtain_bb_grounded(index_bounding_box_first, index_bounding_box_second, relation_type, relation_object_first, relation_object_second)
                            self.log_file.write(f"    Relation processed successfully\n")
                        else:
                            self.log_file.write(f"    No bounding boxes found for this relation\n")
                    else:
                        self.log_file.write(f"    Invalid relation format: {relation}\n")
                    
                    self.log_file.write("\n")
            
            self.log_file.write(f"Total relations processed: {relation_count}\n\n")

        except Exception as e:
            self.log_file.write(f"ERROR in process_relations: {e}\n\n")
            rospy.logwarn(f"Could not process relations: {e}")



    def run_traditional_detection(self, image, depth, scene):
        """Run traditional object detection pipeline"""
        self.start_phase_timer("Traditional Object Detection")
        
        object_relations = scene["Objects"]
        
        # Time label extraction
        label_extraction_start = time.time()
        labels = self.get_classes(object_relations)

        label_extraction_duration = time.time() - label_extraction_start
        self.time_log_file.write(f"  Label Extraction Duration: {label_extraction_duration:.2f} seconds\n")
        
        self.log_file.write("Object Relations from Scene:\n")
        for i, obj in enumerate(object_relations):
            self.log_file.write(f"  {i+1}. {obj}\n")
        self.log_file.write("\n")
        
        self.log_file.write("Extracted Labels for Detection:\n")
        for i, label in enumerate(labels):
            self.log_file.write(f"  {i+1}. {label}\n")
        self.log_file.write("\n")
        
        print("Detected labels: ", labels)
        
        # Time YOLO inference
        yolo_start = time.time()
        self.loader_instance.yolow_model.set_classes(labels)
        bboxs, labels_idx, scores = self.loader_instance.yolow_model.predict(image)
        yolo_duration = time.time() - yolo_start
        self.time_log_file.write(f"  YOLO Inference Duration: {yolo_duration:.2f} seconds\n")
        
        self.log_file.write("YOLO Inference Results:\n")
        self.log_file.write(f"Number of detections: {len(bboxs)}\n")
        self.log_file.write("Bounding boxes: " + str(bboxs) + "\n")
        self.log_file.write("Labels indices: " + str(labels_idx) + "\n")
        self.log_file.write("Scores: " + str(scores) + "\n\n")
        
        # Time detection processing
        detection_processing_start = time.time()
        
        # Process detections
        masked_image = image.copy()
        image_with_bbox = image.copy()
        image_yolow = image.copy()
        self.dict_detections = {}
        overlay_ = masked_image

        index_detection = 0
        print("YOLO inference results:")
        print("bboxs: ", bboxs)
        print("scores: ", scores)
        print("labels_idx: ", labels_idx)
        
        self.log_file.write("Processing Detections (score > 0.1):\n")
        
        for i, (bbox, score, cls_id) in enumerate(zip(bboxs, scores, labels_idx)):
            x1, y1, x2, y2 = bbox

            if score > 0.1:
                label = labels_idx[i]
                self.log_file.write(f"Detection {index_detection + 1}:\n")
                self.log_file.write(f"  Label: {label}\n")
                self.log_file.write(f"  Score: {score:.3f}\n")
                self.log_file.write(f"  Bbox: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})\n")
                
                masks, _ = self.loader_instance.vit_sam_model(masked_image, bbox)
                if index_detection not in self.dict_detections.keys():
                    self.dict_detections[index_detection] = {'bbox': None, 'label': None}
                    self.dict_detections[index_detection]['bbox'] = bbox
                    self.dict_detections[index_detection]['label'] = label
                    cv2.rectangle(image_yolow, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
                    cv2.putText(image_yolow, f"{label}: {score:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 1)

                # Convert binary mask to 3-channel image
                mask_count = 0
                for mask in masks:
                    binary_mask = self.show_mask(mask)
                    overlay = masked_image
                    self.dict_detections[index_detection]['mask'] = binary_mask
                    overlay = cv2.addWeighted(overlay, 1, binary_mask, 0.5, 0)
                    mask_file = os.path.join(self.save_folder, f"rgb_{index_detection}.jpg")
                    cv2.imwrite(mask_file, overlay)
                    overlay_ = cv2.addWeighted(overlay_, 1, binary_mask, 0.5, 0)
                    mask_count += 1
                
                self.log_file.write(f"  Masks generated: {mask_count}\n")
                self.log_file.write(f"  Mask file: rgb_{index_detection}.jpg\n\n")
                index_detection += 1
            else:
                self.log_file.write(f"Detection {i + 1} skipped (score: {score:.3f} < 0.1)\n")

        self.data_reordered = self.dict_detections.copy()
        
        detection_processing_duration = time.time() - detection_processing_start
        self.time_log_file.write(f"  Detection Processing Duration: {detection_processing_duration:.2f} seconds\n")
        
        self.log_file.write(f"Total valid detections: {len(self.dict_detections)}\n\n")
        
        # Time relation processing
        relation_processing_start = time.time()
        self.log_file.write("Processing Object Relations:\n")
        self.process_relations(scene["Objects"])
        relation_processing_duration = time.time() - relation_processing_start
        self.time_log_file.write(f"  Relation Processing Duration: {relation_processing_duration:.2f} seconds\n")
        
        # Time image saving
        image_saving_start = time.time()
        
        # Save detection results
        bbox_count = 0
        for i, value in self.data_reordered.items():
            cv2.rectangle(image_with_bbox, (int(value['bbox'][0]), int(value['bbox'][1])), (int(value['bbox'][2]), int(value['bbox'][3])), (255, 0, 0), 1)
            cv2.putText(image_with_bbox, value['label'], (int(value['bbox'][0] + 10), int(value['bbox'][1] + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            bbox_count += 1

        yolow_path = os.path.join(self.save_folder, "yolow.jpg")
        overlay_path = os.path.join(self.save_folder, "overlay.jpg")
        bbox_path = os.path.join(self.save_folder, "bbox.jpg")
        
        cv2.imwrite(yolow_path, image_yolow)
        cv2.imwrite(overlay_path, overlay_)
        cv2.imwrite(bbox_path, image_with_bbox)
        
        image_saving_duration = time.time() - image_saving_start
        self.time_log_file.write(f"  Image Saving Duration: {image_saving_duration:.2f} seconds\n")
        
        self.log_file.write("Detection Images Saved:\n")
        self.log_file.write(f"  YOLO detections: {yolow_path}\n")
        self.log_file.write(f"  Overlay image: {overlay_path}\n")
        self.log_file.write(f"  Bbox image: {bbox_path}\n\n")
        
        print("Detection images saved.")
        print("Pipeline finished.")

        # Time point cloud coloring
        pcl_coloring_start = time.time()
        self.log_file.write("Coloring Point Cloud with Detections...\n")
        self.color_pcl(image, depth, self.data_reordered)
        pcl_coloring_duration = time.time() - pcl_coloring_start
        self.time_log_file.write(f"  Point Cloud Coloring Duration: {pcl_coloring_duration:.2f} seconds\n")
        
        self.log_file.write("Point cloud coloring completed.\n")
        self.log_file.write("-" * 35 + "\n\n")
        
        self.end_phase_timer("Traditional Object Detection")



