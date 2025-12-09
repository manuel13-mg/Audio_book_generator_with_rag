// DOM Elements
const browseBtn = document.getElementById('browse-btn');
const fileInput = document.getElementById('file-input');
const uploadState = document.getElementById('upload-state');
const loadingState = document.getElementById('loading-state');
const playerState = document.getElementById('player-state');
const progressBar = document.getElementById('progress-bar');
const loadingText = document.getElementById('loading-text');

// Chat Elements
const chatPlaceholder = document.getElementById('chat-placeholder');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const chatHistory = document.getElementById('chat-history');
const ragIndicator = document.getElementById('rag-indicator');
const ragText = document.getElementById('rag-text');

// Audio Element (Created dynamically)
let audioPlayer = new Audio();

// --- Event Listeners ---

browseBtn.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        uploadFile(file);
    }
});

// Chat Send Logic
sendBtn.addEventListener('click', sendChatMessage);
chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendChatMessage();
});

// --- Main Functions ---

async function uploadFile(file) {
    // 1. UI Switch
    uploadState.classList.add('hidden');
    loadingState.classList.remove('hidden');
    loadingText.innerText = "Uploading & Processing...";
    progressBar.style.width = "30%";

    // 2. Prepare FormData
    const formData = new FormData();
    formData.append('file', file);

    try {
        // 3. Send to Flask Backend
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Processing failed');

        const data = await response.json();
        
        // 4. Success!
        progressBar.style.width = "100%";
        loadingText.innerText = "Finalizing...";
        
        setTimeout(() => {
            activateInterface(data);
        }, 500);

    } catch (error) {
        console.error(error);
        loadingText.innerText = "Error: " + error.message;
        loadingText.classList.add('text-red-500');
    }
}

function activateInterface(data) {
    loadingState.classList.add('hidden');
    playerState.classList.remove('hidden');

    // Setup Audio
    document.getElementById('book-title').innerText = data.book_title;
    audioPlayer.src = data.audio_url;
    audioPlayer.play();

    // Activate Chat
    chatPlaceholder.classList.add('hidden');
    chatInput.disabled = false;
    sendBtn.disabled = false;
    document.getElementById('chat-status').innerText = "Ready to discuss";
    
    ragIndicator.classList.remove('bg-gray-500');
    ragIndicator.classList.add('bg-green-500', 'animate-pulse');
    ragText.innerText = "RAG Active";
    ragText.classList.add('text-green-500');
}

async function sendChatMessage() {
    const text = chatInput.value.trim();
    if (!text) return;

    // Add User Message to UI
    appendMessage('user', text);
    chatInput.value = '';

    try {
        // Send to Flask RAG endpoint
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: text })
        });

        const data = await response.json();
        appendMessage('ai', data.answer);

    } catch (error) {
        appendMessage('ai', "Sorry, I couldn't reach the server.");
    }
}

function appendMessage(sender, text) {
    const div = document.createElement('div');
    // Simple logic to style based on sender
    if (sender === 'user') {
        div.className = "flex justify-end mb-4";
        div.innerHTML = `<div class="bg-indigo-600 text-white rounded-l-lg rounded-tr-lg p-3 max-w-md"><p>${text}</p></div>`;
    } else {
        div.className = "flex justify-start mb-4";
        div.innerHTML = `<div class="bg-gray-800 text-gray-200 rounded-r-lg rounded-tl-lg p-3 max-w-md shadow-sm"><p>${text}</p></div>`;
    }
    chatHistory.appendChild(div);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}