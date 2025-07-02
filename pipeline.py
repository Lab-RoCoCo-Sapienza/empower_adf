#!/usr/bin/env python

import math
import random
import rospy
from utils.config import *
from scene_acquisition import *
from agents import *
import pickle
from matplotlib.colors import to_rgb
import open3d as o3d
import os
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import numpy as np
import tf2_ros

rospy.init_node('empower_node', anonymous=True)


class Pipeline():
    def __init__(self):
        self.loader_instance = None
        self.COLORS = ['red', 'green', 'blue', 'magenta', 'cyan', 'yellow']*3
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        self.publisher_centroid = rospy.Publisher("/pcl_centroids", MarkerArray, queue_size=100)
        self.publisher_maximum = rospy.Publisher("/pcl_maximum", MarkerArray, queue_size=100)
        self.publisher_names = rospy.Publisher("/pcl_names", MarkerArray, queue_size=100)



    def rgb_to_bgr(self, rgb_color):
        r, g, b = rgb_color
        return [b, g, r]

    def set_loader(self, loader_instance):
        self.loader_instance = loader_instance

    def compare_two_words(self,list1, list2):
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
        sim_word =  np.mean(sim_word)
        return sim_word

    def is_in_list(self,word,list):
        for obj in list:
            if self.compare_two_words(word, obj) != None and self.compare_two_words(word, obj) > 0.708:
                return True
        return False
    
    def get_classes(self,object_relations):
        relation_list = []

        for relation in object_relations:

            relation = relation.replace("(", "")
            relation_object_first = relation.split(")")[1].split(",")[0]
            relation_object_second = relation.split(")")[1].split(",")[2]
            word_1 = self.split_word(relation_object_first)
            word_2 = self.split_word(relation_object_second)
            if not self.is_in_list(word_1,relation_list):
                relation_list.append(word_1)
            if not self.is_in_list(word_2,relation_list):
                relation_list.append(word_2)
        for index in range(len(relation_list)):
            relation_list[index] = " ".join(relation_list[index])
        return relation_list
    
    def compare_two_list_of_objects(self,position_in_image_first,position_in_image_second,relation,object_first,object_second):
        if position_in_image_first == {} or position_in_image_second == {}:
            return
    
        if "on" in relation:
            min_distance_x = 30000
            for key_first in position_in_image_first.keys():
                for key_second in position_in_image_second.keys():
                    dis_x = abs(position_in_image_first[key_first]['x'] - position_in_image_second[key_second]['x'])
                    dis_y = abs(position_in_image_first[key_first]['y'] - position_in_image_second[key_second]['y'])
                    distance = math.sqrt((dis_x*dis_x + dis_y*dis_y))
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
                    min_distance_x_bb = abs(position_in_image_first[key_first]['x'] - position_in_image_second[key_second]['x'])//2
                    min_distance_y_bb = abs(position_in_image_first[key_first]['y'] - position_in_image_second[key_second]['y'])//2
                    if min_distance_x_bb < min_distance_x and min_distance_x_bb != 0 and min_distance_y > min_distance_y_bb :
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
                    min_distance_x_bb = abs(position_in_image_first[key_first]['x'] - position_in_image_second[key_second]['x'])//2
                    min_distance_y_bb = abs(position_in_image_first[key_first]['y'] - position_in_image_second[key_second]['y'])//2
                    if min_distance_x_bb < min_distance_x and min_distance_x_bb != 0 and min_distance_y > min_distance_y_bb and min_distance_y_bb != 0:
                        
                        index_first = key_first
                        index_second = key_second
                        min_distance_x = min_distance_x_bb
                        min_distance_y = min_distance_y_bb
            self.data_reordered[index_first]['label'] = object_first
            self.data_reordered[index_second]['label'] = object_second
            self.dict_detections.pop(index_first)
            self.dict_detections.pop(index_second)

    def obtain_bb_grounded(self,index_first,index_second,relation,object_first,object_second):
            detection_data = self.dict_detections
            position_in_image_first = {}
            position_in_image_second = {}
            for i in range(len(index_first)):
                if index_first[i] not in position_in_image_first.keys():
                    position_in_image_first[index_first[i]] = {'x':None,'y':None}
                position_in_image_first[index_first[i]]['x'] = (detection_data[index_first[i]]['bbox'][0] + detection_data[index_first[i]]['bbox'][2]) //2
                position_in_image_first[index_first[i]]['y'] = (detection_data[index_first[i]]['bbox'][1] + detection_data[index_first[i]]['bbox'][3]) //2
            for i in range(len(index_second)):
                if index_second[i] not in position_in_image_second.keys():
                    position_in_image_second[index_second[i]] = {'x':None,'y':None}
                position_in_image_second[index_second[i]]['x'] = (detection_data[index_second[i]]['bbox'][0] + detection_data[index_second[i]]['bbox'][2]) //2
                position_in_image_second[index_second[i]]['y'] = (detection_data[index_second[i]]['bbox'][1] + detection_data[index_second[i]]['bbox'][3]) //2
            self.compare_two_list_of_objects(position_in_image_first,position_in_image_second,relation,object_first,object_second)
      
    def split_word(self,words):
        splitted_word = []
        words = words.lower()
        doc = self.loader_instance.nlp(words)
        for token in doc:
            if token.pos_ == "AUX" or (token.pos_ == "NOUN" and token.dep_ in ["dobj","ROOT","nsubj"]) or token.pos_ == "VERB" or token.pos_ == "PROPN":
                splitted_word.append(token.text) 
        return splitted_word
    
    def show_mask(self,mask, random_color = True):
        if random_color:
            color = np.concatenate([np.random.random(3)], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        mask_image = mask_image.numpy() * 255
        mask_image = mask_image.astype(np.uint8)
        return mask_image

    def find_bb_relation(self,relation_object):
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
        R = 2*np.array([[pow(qw,2) + pow(qx,2) - 0.5, qx*qy-qw*qz, qw*qy+qx*qz],[qw*qz+qx*qy, pow(qw,2) + pow(qy,2) - 0.5, qy*qz-qw*qx],[qx*qz-qw*qy, qw*qx+qy*qz, pow(qw,2) + pow(qz,2) - 0.5]])
        return R, T

    def set_marker(self, point, color, id, R, T):

        point = point/1000
        transform = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
        point = np.dot(transform, point) #in xtion
        R = np.transpose(R)
        point = np.dot(R, point-T) #in map

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

    def color_pcl(self, image, pcd, detections):
        pcd.colors = o3d.utility.Vector3dVector(np.tile(to_rgb('gray'), (len(pcd.points), 1)))

        h, w, _ = image.shape

        masks = []
        masks_flipped = []

        for key in detections.keys():
            mask = detections[key]['mask']
            masks.append(mask[:,:,0])
            masks_flipped.append(cv2.flip(mask[:,:,0],1))
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
                    image[y, x] = self.rgb_to_bgr([int(color*255) for color in to_rgb(self.COLORS[id_color])])
                
                if 0 <= x < w and 0 <= y < h and mask_flipped[y, x] != 0:
                    if id_color not in colors_dict.keys():
                        colors_dict[id_color] = []
                    pcd.colors[idx] = to_rgb(self.COLORS[id_color])
                    colors_dict[id_color].append(point)
        
        array_centroids = MarkerArray()
        array_maximum = MarkerArray()
        array_name = MarkerArray()
        trans = self.tf_buffer.lookup_transform("xtion_rgb_optical_frame", "map",  rospy.Time(0), rospy.Duration(2.0))
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
            
            # Create unique marker IDs using time prefix, TODO:FIX
            time_id_prefix = random.randint(1,1000)
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
            print("finished")
        self.publisher_centroid.publish(array_centroids)
        self.publisher_maximum.publish(array_maximum)
        self.publisher_names.publish(array_name)
        print("upblished")
        cv2.imwrite(LOG_DIR+'colored_image.png', image)
        o3d.visualization.draw_geometries([pcd])
        return



    def run_pipeline(self):
        # Acquire image
        #image, depth = local_acquire_image("/tiago_public_ws/src/code_planning_awareness/empower/output/20250515-115512/")
        image, depth = acquire_image()
        print("Image acquired")
        
        # Define task
        task = "Pick the TV and laptop"
        print(task)
        
        #ONLINE PROCEDURE        
        enviroment_info,description_agent_info, plan = multi_agent_planning(image, task)

        self.results_multi = {
            "environment_agent_info": enviroment_info,
            "description_agent_info": description_agent_info,
            "planning_agent_info": plan,
        }

        print("Environment agent info: ", enviroment_info)
        print("Description agent info: ", description_agent_info)
        print("Planning agent info: ", plan)

        with open(LOG_DIR + "planning.pkl",'wb') as f:
            pickle.dump(self.results_multi, f, protocol=2)
        
        with open(LOG_DIR + "planning.txt",'w') as f:
            f.write("Environment agent info: ")
            f.write(self.results_multi["environment_agent_info"])
            f.write("\nDescription agent info: ")
            f.write(self.results_multi["description_agent_info"])
            f.write("\nPlanning agent info: ")
            f.write(self.results_multi["planning_agent_info"])
        print("Results saved to file.")
        """


        #OFFLINE PROCEDURE FOR TEST

        with open(LOG_DIR + "planning.pkl",'rb') as f:
            self.results_multi = pickle.load(f)
        print("Results loaded from file.")
        plan= self.results_multi["environment_agent_info"]
        description_agent_info = self.results_multi["description_agent_info"]
        enviroment_info = self.results_multi["planning_agent_info"]

        print("Environment agent info: ", enviroment_info)
        print("Description agent info: ", description_agent_info)
        print("Planning agent info: ", plan)



        """
        object_relations = enviroment_info.split('\n')
        print("object relations: ", object_relations)
        labels = self.get_classes(object_relations)
        #labels = ['tv', 'laptop', 'table']
        print("labels: ", labels)

        # print("end vocabulary : " + str(time.time() -start))

        self.loader_instance.yolow_model.set_classes(labels)
        bboxs, labels_idx, scores = self.loader_instance.yolow_model.predict(image)
        masked_image = image.copy()
        image_with_bbox = image.copy()
        image_yolow = image.copy()
        self.dict_detections = {}
        overlay_ = masked_image 

        index_detection = 0
        #YOLOW inference
        start = time.time()
        print("yolow inference : " + str(time.time() -start))
        print("bboxs: ", bboxs)
        print("scores: ", scores) 
        print("labels_idx: ", labels_idx)
        for i, (bbox,score,cls_id) in enumerate(zip(bboxs, scores, labels_idx)):
            x1,y1,x2,y2 = bbox

            if score > 0.1:

                label = labels_idx[i]
                masks, _ = self.loader_instance.vit_sam_model(masked_image, bbox)
                if index_detection not in self.dict_detections.keys():
                    self.dict_detections[index_detection] = {'bbox':None,'label':None}
                    self.dict_detections[index_detection]['bbox'] = bbox
                    self.dict_detections[index_detection]['label'] = label
                    cv2.rectangle(image_yolow, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
                    cv2.putText(image_yolow, f"{label}: {score:.2f}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 1)
                    
                # Convert binary mask to 3-channel image
                for mask in masks:
                    binary_mask = self.show_mask(mask)
                    overlay = masked_image
                    self.dict_detections[index_detection]['mask'] = binary_mask
                    overlay = cv2.addWeighted(overlay, 1, binary_mask, 0.5, 0)
                    cv2.imwrite(LOG_DIR+f"rgb_{index_detection}.jpg", overlay)
                    overlay_ = cv2.addWeighted(overlay_, 1, binary_mask, 0.5, 0)
                index_detection += 1
        
        self.data_reordered = self.dict_detections.copy()
        # start = time.time()
        for relation in object_relations:
                relation = relation.replace("(", "")

                relation_object_first = relation.split(")")[1].split(",")[0]
                relation_object_first = relation_object_first[1:]
                relation_type = relation.split(")")[1].split(",")[1]
                relation_type = relation_type[1:]
                relation_object_second = relation.split(")")[1].split(",")[2]
                relation_object_second = relation_object_second[1:]
                index_bounding_box_first = self.find_bb_relation(relation_object_first)
                index_bounding_box_second = self.find_bb_relation(relation_object_second)
                if index_bounding_box_first != [] or index_bounding_box_second != []:
                    self.obtain_bb_grounded(index_bounding_box_first,index_bounding_box_second,relation_type,relation_object_first,relation_object_second)

        # print("grounfing : " + str(time.time() -start))
        for i, value in self.data_reordered.items():
            cv2.rectangle(image_with_bbox, (int(value['bbox'][0]), int(value['bbox'][1])), (int(value['bbox'][2]), int(value['bbox'][3])), (255, 0, 0), 1 )
            cv2.putText(image_with_bbox, value['label'], (int(value['bbox'][0]+10), int(value['bbox'][1]+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(LOG_DIR+f"yolow.jpg", image_yolow)
        cv2.imwrite(LOG_DIR+f"overlay.jpg", overlay_)
        cv2.imwrite(LOG_DIR+f"bbox.jpg", image_with_bbox)
        print("Image with bounding boxes saved.")
        print("Overlay image saved.")
        print("Image with masks saved.")
        print("Pipeline finished.")

        self.color_pcl(image, depth, self.data_reordered)

