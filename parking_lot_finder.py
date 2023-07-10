import logging as log
import sys
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
from time import perf_counter
import os
import cv2
sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))
sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python/openvino/model_zoo'))
from model_api.models import DetectionModel, DetectionWithLandmarks, RESIZE_TYPES, OutputTransform
from model_api.performance_metrics import PerformanceMetrics
from model_api.pipelines import get_user_config, AsyncPipeline
from model_api.adapters import create_core, OpenvinoAdapter, OVMSAdapter
import monitors
from images_capture import open_images_capture
from helpers import resolution, log_latency_per_stage
from visualizers import ColorPalette
import sys
import logging as log
from time import perf_counter
from argparse import ArgumentParser, SUPPRESS

from tqdm import tqdm
import numpy as np
import wave
from openvino.runtime import Core, get_version

from models.forward_tacotron_ie import ForwardTacotronIE
from models.mel2wave_ie import WaveRNNIE, MelGANIE
from utils.gui import init_parameters_interactive
from playsound import playsound

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)

def draw_detections(frame, detections, palette, labels, output_transform):
    a=[]
    cnt=0
    k=32
    up=0
    frame = output_transform.resize(frame)
    xlocation_1=[240,210,100,30]
    ylocation_1=[50,165,405,550]
    # f=open("C:/Users/AIoT19/open_model/open_model_zoo/demos/text_to_speech_demo/python/text-to-speech_test.txt",'w')
    for detection in detections:
        k=32        
        class_id = int(detection.id)
        color = palette[class_id]
        det_label = labels[class_id] if labels and len(labels) >= class_id else '#{}'.format(class_id)
        xmin, ymin, xmax, ymax = detection.get_coords()
        xmin, ymin, xmax, ymax = output_transform.scale([xmin, ymin, xmax, ymax])
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        for i in range(0,2):
            up=0
            up_2=0
            for j in range(0,8):
                cv2.rectangle(frame, (xlocation_1[i]+up+up_2, ylocation_1[i]), (xlocation_1[i]+90+up+up_2, ylocation_1[i]+85), (0,255,255), 2)               

                if xlocation_1[i]+up+up_2< int((xmax+xmin)/2) and xlocation_1[i]+80+up+up_2 >int((xmax+xmin)/2) and ylocation_1[i] <int((ymax+ymin)/2) and ylocation_1[i]+85>int((ymax+ymin)/2):
                    cv2.circle(frame,(int((xmax+xmin)/2)+3,int((ymax+ymin)/2)+3),3,(0,0,255),2)                    
                    a.append(xmin)                    
                up+=130
                if i==0 and j>=3:
                    up_2-=10

        for q in range(2,len(xlocation_1)):
            up=0
            up_1=0

            for l in range(0,8):
                cv2.rectangle(frame, (xlocation_1[q]+up+up_1, ylocation_1[q]), (xlocation_1[q]+90+up+up_1, ylocation_1[q]+100), (0,255,255), 2)               

                if xlocation_1[q]+up+up_1< int((xmax+xmin)/2) and xlocation_1[q]+90+up+up_1 >int((xmax+xmin)/2) and ylocation_1[q] <int((ymax+ymin)/2) and ylocation_1[q]+100>int((ymax+ymin)/2):
                    cv2.circle(frame,(int((xmax+xmin)/2)+3,int((ymax+ymin)/2)+3),3,(0,0,255),2)                    
                    a.append(xmin)                    
                up+=135
                up_1+=10
                if q==3 and l>=3:
                    up_1+=15     
        
        if isinstance(detection, DetectionWithLandmarks):
            for landmark in detection.landmarks:
                landmark = output_transform.scale(landmark)
                cv2.circle(frame, (int(landmark[0]), int(landmark[1])), 2, (0, 255, 255), 2)       
        cnt=len(a)        
        f=open("C:/Users/AIoT19/open_model/open_model_zoo/demos/text_to_speech_demo/python/text-to-speech.txt",'w')
        k=k-cnt             
        # print(k)
        if k==0:
            f.write("The parking lot is full.")     
        elif k==32:
            f.write("All parking areas are empty.")         
        else:
            f.write("we have 32 site in the parking lot, but There are only "+str(k)+" placess left.")
            
   
        f.close() 

    return frame,cnt

def print_raw_results(detections, labels, frame_id):
    log.debug(' ------------------- Frame # {} ------------------ '.format(frame_id))
    log.debug(' Class ID | Confidence | XMIN | YMIN | XMAX | YMAX ')
    for detection in detections:
        xmin, ymin, xmax, ymax = detection.get_coords()
        class_id = int(detection.id)
        det_label = labels[class_id] if labels and len(labels) >= class_id else '#{}'.format(class_id)
        log.debug('{:^9} | {:10f} | {:4} | {:4} | {:4} | {:4} '
                  .format(det_label, detection.score, xmin, ymin, xmax, ymax))

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)


def save_wav(x, path):
    sr = 22050
    try:

        with wave.open(path, "w") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(sr)
            f.writeframes(x.tobytes())
        os.system('C:/Users/AIoT19/open_model/open_model_zoo/demos/text_to_speech_demo/python/res.wav')
    except Exception:
        pass
def is_correct_args(args):
    if not ((args.model_melgan is None and args.model_rnn is not None and args.model_upsample is not None) or
            (args.model_melgan is not None and args.model_rnn is None and args.model_upsample is None)):
        log.error('Can not use m_rnn and m_upsample with m_melgan. Define m_melgan or [m_rnn, m_upsample]')
        return False
    if args.alpha < 0.5 or args.alpha > 2.0:
        log.error('Can not use time coefficient less than 0.5 or greater than 2.0')
        return False
    if args.speaker_id < -1 or args.speaker_id > 39:
        log.error('Mistake in the range of args.speaker_id. Speaker_id should be -1 (GUI regime) or in range [0,39]')
        return False

    return True

def parse_input(input):
    if not input:
        return
    sentences = []
    for text in input:
        if text.endswith('.txt'):
            try:
                with open(text, 'r', encoding='utf8') as f:
                    sentences += f.readlines()
                continue
            except OSError:
                pass
        sentences.append(text)
    return sentences

def main():
    k=32
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    
    for i in range(1,6): 
        args.model="C:/Users/AIoT19/open_model/open_model_zoo/demos/object_detection_demo/python/intel/vehicle-detection-0202/FP16/vehicle-detection-0202.xml "
        args.architecture_type='ssd'
        args.adapter="openvino"
        # args.input="C:/Users/AIoT19/Desktop/img.png"
        args.input="C:/Users/AIoT19/Desktop/img{}.png".format(i)
        args.device='CPU'
        args.labels= None
        args.prob_threshold=0.5
        args.resize_type= None
        args.input_size=(600,600)
        args.anchors=None
        args.masks=None
        args.layout=None
        args.num_classes=None
        args.num_infer_requests=0
        args.num_streams=''
        args.num_threads=None
        args.loop=False
        args.output="C:/Users/AIoT19/Desktop/test_0706_reuslt.png"
        args.no_show=None
        args.output_resolution=None
        args.utilization_monitors=''
        args.resize_type=None
        args.mean_values=None
        args.scale_values=None
        args.reverse_input_channels=False
        args.raw_output_message=False
        args.output_limit=1000

        if args.architecture_type != 'yolov4' and args.anchors:
            log.warning('The "--anchors" option works only for "-at==yolov4". Option will be omitted')
        if args.architecture_type != 'yolov4' and args.masks:
            log.warning('The "--masks" option works only for "-at==yolov4". Option will be omitted')
        if args.architecture_type not in ['nanodet', 'nanodet-plus'] and args.num_classes:
            log.warning('The "--num_classes" option works only for "-at==nanodet" and "-at==nanodet-plus". Option will be omitted')
        
        cap = open_images_capture(args.input, args.loop)

        if args.adapter == 'openvino':
            plugin_config = get_user_config(args.device, args.num_streams, args.num_threads)
            model_adapter = OpenvinoAdapter(create_core(), args.model, device=args.device, plugin_config=plugin_config,
                                            max_num_requests=args.num_infer_requests, model_parameters = {'input_layouts': args.layout})
        elif args.adapter == 'ovms':
            model_adapter = OVMSAdapter(args.model)

        configuration = {
            'resize_type': args.resize_type,
            'mean_values': args.mean_values,
            'scale_values': args.scale_values,
            'reverse_input_channels': args.reverse_input_channels,
            'path_to_labels': args.labels,
            'confidence_threshold': args.prob_threshold,
            'input_size': args.input_size, # The CTPN specific
            'num_classes': args.num_classes, # The NanoDet and NanoDetPlus specific
        }
        model = DetectionModel.create_model(args.architecture_type, model_adapter, configuration)
        model.log_layers_info()

        detector_pipeline = AsyncPipeline(model)

        next_frame_id = 0
        next_frame_id_to_show = 0

        palette = ColorPalette(len(model.labels) if model.labels else 100)
        metrics = PerformanceMetrics()
        render_metrics = PerformanceMetrics()
        presenter = None
        output_transform = None
        video_writer = cv2.VideoWriter()

        while True:
            if detector_pipeline.callback_exceptions:
                raise detector_pipeline.callback_exceptions[0]
            # Process all completed requests
            results = detector_pipeline.get_result(next_frame_id_to_show)
            if results:
                objects, frame_meta = results
                frame = frame_meta['frame']
                start_time = frame_meta['start_time']

                if len(objects) and args.raw_output_message:
                    print_raw_results(objects, model.labels, next_frame_id_to_show)

                presenter.drawGraphs(frame)
                rendering_start_time = perf_counter()
                frame= draw_detections(frame, objects, palette, model.labels, output_transform)
                # cv2.putText("result",f"{num_1}",(50,50),cv2.FONT_HERSHEY_SIMPLEX,
                #         0.8,
                #         (255,255,255),
                #         1,
                #         cv2.LINE_AA,)
                
                render_metrics.update(rendering_start_time)
                metrics.update(start_time, frame)

                if video_writer.isOpened() and (args.output_limit <= 0 or next_frame_id_to_show <= args.output_limit-1):
                    video_writer.write(frame)
                next_frame_id_to_show += 1

                if not args.no_show:
                    
                    cv2.imshow('Detection Results', frame)
                    key = cv2.waitKey(0)

                    ESC_KEY = 27
                    # Quit.
                    if key in {ord('q'), ord('Q'), ESC_KEY}:
                        break
                    presenter.handleKey(key)
                continue

            if detector_pipeline.is_ready():
                # Get new image/frame
                start_time = perf_counter()
                frame = cap.read()
                if frame is None:
                    if next_frame_id == 0:
                        raise ValueError("Can't read an image from the input")
                    break
                if next_frame_id == 0:
                    output_transform = OutputTransform(frame.shape[:2], args.output_resolution)
                    if args.output_resolution:
                        output_resolution = output_transform.new_resolution
                    else:
                        output_resolution = (frame.shape[1], frame.shape[0])
                    presenter = monitors.Presenter(args.utilization_monitors, 55,
                                                (round(output_resolution[0] / 4), round(output_resolution[1] / 8)))
                    if args.output and not video_writer.open(args.output, cv2.VideoWriter_fourcc(*'MJPG'),
                                                            cap.fps(), output_resolution):
                        raise RuntimeError("Can't open video writer")
                # Submit for inference
                detector_pipeline.submit_data(frame, next_frame_id, {'frame': frame, 'start_time': start_time})
                next_frame_id += 1
            else:
                # Wait for empty request
                detector_pipeline.await_any()

        detector_pipeline.await_all()
        if detector_pipeline.callback_exceptions:
            raise detector_pipeline.callback_exceptions[0]
        # Process completed requests
        for next_frame_id_to_show in range(next_frame_id_to_show, next_frame_id):
            results = detector_pipeline.get_result(next_frame_id_to_show)
            objects, frame_meta = results
            frame = frame_meta['frame']
            start_time = frame_meta['start_time']

            if len(objects) and args.raw_output_message:
                print_raw_results(objects, model.labels, next_frame_id_to_show)

            presenter.drawGraphs(frame)
            rendering_start_time = perf_counter()
            frame,cnt_1= draw_detections(frame, objects, palette, model.labels, output_transform)
            cv2.putText(frame,f'{k-cnt_1}',(50,100),cv2.FONT_HERSHEY_SIMPLEX,
                        3,
                        (0,0,0),
                        3,)
            
            
            


            
            render_metrics.update(rendering_start_time)
            metrics.update(start_time, frame)

            if video_writer.isOpened() and (args.output_limit <= 0 or next_frame_id_to_show <= args.output_limit-1):
                video_writer.write(frame)

            if not args.no_show:
                cv2.imshow('Detection Results', frame)
                key = cv2.waitKey(0)

                ESC_KEY = 27
                # Quit.
                if key in {ord('q'), ord('Q'), ESC_KEY}:
                    break
                presenter.handleKey(key)

        metrics.log_total()
        log_latency_per_stage(cap.reader_metrics.get_latency(),
                            detector_pipeline.preprocess_metrics.get_latency(),
                            detector_pipeline.inference_metrics.get_latency(),
                            detector_pipeline.postprocess_metrics.get_latency(),
                            render_metrics.get_latency())
        for rep in presenter.reportMeans():
            log.info(rep)




        
        parser = ArgumentParser(add_help=False)
        args = parser.add_argument_group('Options')
        args.model_melgan = 'C:/Users/AIoT19/open_model/open_model_zoo/demos/text_to_speech_demo/python/intel/text-to-speech-en-0001/text-to-speech-en-0001-generation/FP16/text-to-speech-en-0001-generation.xml'
        args.model_duration = 'C:/Users/AIoT19/open_model/open_model_zoo/demos/text_to_speech_demo/python/intel/text-to-speech-en-0001/text-to-speech-en-0001-duration-prediction/FP16/text-to-speech-en-0001-duration-prediction.xml'
        args.model_forward = 'C:/Users/AIoT19/open_model/open_model_zoo/demos/text_to_speech_demo/python/intel/text-to-speech-en-0001/text-to-speech-en-0001-regression/FP16/text-to-speech-en-0001-regression.xml'
        args.out = 'C:/Users/AIoT19/open_model/open_model_zoo/demos/text_to_speech_demo/python/res{}.wav'.format(i)
        args.input = ['C:/Users/AIoT19/open_model/open_model_zoo/demos/text_to_speech_demo/python/text-to-speech.txt']
        args.device = 'CPU'
        args.alpha = 1.0
        args.speaker_id = 19
        args.upsampler_width = -1
        args.model_upsample = None
        args.model_rnn = None

        if not is_correct_args(args):
            return 1

        log.info('OpenVINO Runtime')
        log.info('\tbuild: {}'.format(get_version()))
        core = Core()

        if args.model_melgan is not None:
            vocoder = MelGANIE(args.model_melgan, core, device=args.device)
        else:
            vocoder = WaveRNNIE(args.model_upsample, args.model_rnn, core, device=args.device,
                                upsampler_width=args.upsampler_width)

        forward_tacotron = ForwardTacotronIE(args.model_duration, args.model_forward, core, args.device, verbose=False)

        audio_res = np.array([], dtype=np.int16)

        speaker_emb = None
        if forward_tacotron.is_multi_speaker:
            if args.speaker_id == -1:
                interactive_parameter = init_parameters_interactive(args)
                args.alpha = 1.0 / interactive_parameter["speed"]
                speaker_emb = forward_tacotron.get_pca_speaker_embedding(interactive_parameter["gender"],
                                                                        interactive_parameter["style"])
            else:
                speaker_emb = [forward_tacotron.get_speaker_embeddings()[args.speaker_id, :]]
        len_th = 80
        input_data = parse_input(args.input)
        time_forward = 0
        time_wavernn = 0
        time_s_all = perf_counter()
        count = 0
        for line in input_data:
            count += 1
            line = line.rstrip()
            log.info("Process line {0} with length {1}.".format(count, len(line)))

            if len(line) > len_th:
                texts = []
                prev_begin = 0
                delimiters = '.!?;:,'
                for i, c in enumerate(line):
                    if (c in delimiters and i - prev_begin > len_th) or i == len(line) - 1:
                        texts.append(line[prev_begin:i + 1])
                        prev_begin = i + 1
            else:
                texts = [line]

            for text in tqdm(texts):
                time_s = perf_counter()
                mel = forward_tacotron.forward(text, alpha=args.alpha, speaker_emb=speaker_emb)
                time_forward += perf_counter() - time_s

                time_s = perf_counter()
                audio = vocoder.forward(mel)
                time_wavernn += perf_counter() - time_s

                audio_res = np.append(audio_res, audio)

        total_latency = (perf_counter() - time_s_all) * 1e3
        log.info("Metrics report:")
        log.info("\tLatency: {:.1f} ms".format(total_latency))
        log.debug("\tVocoder time: {:.1f} ms".format(time_wavernn * 1e3))
        log.debug("\tForwardTacotronTime: {:.1f} ms".format(time_forward * 1e3))
        save_wav(audio_res, args.out)
        playsound('C:/Users/user/openvino/open_model_zoo/demos/text_to_speech_demo/python/res{}.wav'.format(i))
        



if __name__ == '__main__':
    sys.exit(main() or 0)







