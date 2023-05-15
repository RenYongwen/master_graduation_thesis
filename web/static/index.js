const audioFile = document.getElementById("audio-file");
const fileInput = document.querySelector('.file-input');
const fileName = document.querySelector('.file-name');

// 录制音频
let stream, recorder, chunks = [];
function toggleRecording(record) {
    if (record.innerHTML === '录制') {
        startRecording();
    } else {
        stopRecording();
    }
}
const startRecording = () => {
    record.innerHTML = '停止';
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(str => {
            stream = str;
            recorder = new MediaRecorder(stream);
            recorder.addEventListener('dataavailable', e => chunks.push(e.data));
            recorder.start();
        });
};
const stopRecording = () => {
    record.innerHTML = '录制';
    recorder.stop();
    stream.getTracks().forEach(track => track.stop());
    const blob = new Blob(chunks, { type: 'audio/wav' });
    const url = URL.createObjectURL(blob);
    audioFile.src = url;
};

// 输入
fileInput.addEventListener('change', function () {
    if (fileInput.value) {
        console.log(fileInput.value.split('\\').pop());
        fileName.textContent = fileInput.value.split('\\').pop();
        const url = URL.createObjectURL(fileInput.files[0]);
        audioFile.src = url;
    } else {
        fileName.textContent = '没有选择文件';
    }
});

// 预测
function predict() {
    const formData = new FormData();
    fetch(audioFile.src)
        .then(response => response.arrayBuffer())
        .then(buffer => {
            // 处理获取到的二进制数据
            const file = new File([buffer], 'audio.wav', { type: 'audio/wav' });;
            formData.append("audio", file);
            console.log(file);
        })
        .then(()=>{
            return fetch("/predict", {
                method: "POST",
                body: formData
            });
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById('results').innerHTML = '';
            const class_probs = data;
            for (let i = 0; i < 5; i++) {
                const liContent = `<div style="display: flex; justify-content: space-between;"><p>${class_probs[i].class}</p><p>${class_probs[i].prob.toFixed(5)}</p></div><div class="progress-bar"><div class="progress" style="width: ${class_probs[i].prob * 100}%"></div></div>`;
                const li = document.createElement('li');
                li.innerHTML = liContent;
                document.getElementById('results').appendChild(li);
            }
        })
        .catch(error => {
            document.getElementById('results').innerHTML = '预测出错！！！';
            console.error(error);
        });
}
