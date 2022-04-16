import sounddevice as sd
from scipy.io.wavfile import write
import time 

def sounddeviceee(tiempo:int):
    fs = 16000 
    seconds = tiempo 

    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait() 
    write('./test/dog/output.wav', fs, myrecording) 

def main(tiempo:int):
    print("Va a grabar un sonido de un perro en 3 segundos")
    time.sleep(1)
    print("3....")
    time.sleep(1)
    print("2....")
    time.sleep(1)
    print("1....")
    sounddeviceee(tiempo=tiempo)
    print("Listo!!")
main(10)
