import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import speech_recognition as sr


class VoiceCommandNode(Node):
    def __init__(self):
        super().__init__('qwen_robot_voice')

        self.pub = self.create_publisher(String, '/qwen_robot/command', 10)

        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        self.get_logger().info('Qwen Robot voice node started.')
        self.get_logger().info('Listening and publishing to /qwen_robot/command')

        self.timer = self.create_timer(0.1, self.listen_once)
        self.listening = False

    def listen_once(self):
        if self.listening:
            return

        self.listening = True

        try:
            with self.microphone as source:
                self.get_logger().info('Listening...')
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=4)

            text = self.recognizer.recognize_google(audio)
            text = text.lower().strip()

            self.get_logger().info(f'Heard: {text}')

            msg = String()
            msg.data = text
            self.pub.publish(msg)

        except sr.WaitTimeoutError:
            pass

        except sr.UnknownValueError:
            self.get_logger().warn('Could not understand audio.')

        except Exception as e:
            self.get_logger().error(str(e))

        self.listening = False


def main(args=None):
    rclpy.init(args=args)
    node = VoiceCommandNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
