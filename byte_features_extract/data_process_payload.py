__author__ = 'HPC'
import os
import glob
import binascii

import scapy.all as scapy
from tqdm import tqdm
import numpy as np
import torch
from torchvision import transforms
from scipy.ndimage import zoom

def makedir(path):
    try:
        os.mkdir(path)
    except Exception as E:
        pass


def read_Grayscale_Image_bytes(pcap_dir):
    packets = scapy.rdpcap(pcap_dir)  
    headers = []  
    payloads = []  
    
    for packet in packets:
        header = (binascii.hexlify(bytes(packet['IP']))).decode()  
        try:
            payload = (binascii.hexlify(bytes(packet['Raw']))).decode()  
            header = header.replace(payload, '')  
        except:
            payload = ''
        
        
        if len(header) > 160:
            header = header[:160]  
        elif len(header) < 160:
            header += '0' * (160 - len(header))  
        
        if len(payload) > 640:
            payload = payload[:640]  
        elif len(payload) < 640:
            payload += '0' * (640 - len(payload))  
        
        headers.append(header)  
        payloads.append(payload)  
        
        if len(headers) >= 5:  
            break
            
    
    while len(headers) < 5:
        headers.append('0' * 160)  
        payloads.append('0' * 640)  

    
    header_matrix = np.array([list(int(header[i:i + 2], 16) for i in range(0, len(header), 2)) for header in headers]).reshape(20, 20)
    
    header_matrix = zoom(header_matrix, zoom=2, order=1)  
    
    
    payload_matrix = np.array([list(int(payload[i:i + 2], 16) for i in range(0, len(payload), 2)) for payload in payloads]).reshape(40, 40)
    
    fh = np.vstack((header_matrix, payload_matrix))
    
    return payload_matrix  

def Grayscale_Image_generator(flows_pcap_path, output_path):
    if not os.path.exists(output_path):
        
        os.makedirs(output_path)
    items = os.listdir(output_path)
    if len(items) != 0:
        return
    flows = glob.glob(flows_pcap_path + "/*/*.pcap")  
    classes = glob.glob(flows_pcap_path + "/*/*")  
    
    for cla in tqdm(classes):  
        makedir(cla.replace(flows_pcap_path, output_path))  
        
    for flow in tqdm(flows):  
        fh = read_Grayscale_Image_bytes(flow)  
        
        fh = np.uint8(fh)  
        to_pil = transforms.ToPILImage()  
        im = to_pil(torch.from_numpy(fh))  
        
        output_file = flow.replace('.pcap', '.png').replace(flows_pcap_path, output_path)  
        output_dir = os.path.dirname(output_file)  
        os.makedirs(output_dir, exist_ok=True)  
        im.save(output_file)  
    print("载荷特征矩阵构建完成")

