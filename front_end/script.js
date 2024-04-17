function sendMessage() {
    var userInput = document.getElementById("user-input").value;
    if (userInput.trim() === "") {
        return;
    }

    var chatBox = document.getElementById("chat-box");
    var userMessage = '<div class="user-message">' + userInput + '</div>';
    chatBox.innerHTML += userMessage;

    // 这里模拟 ChatGPT 的响应，实际上应该是通过 AJAX 请求发送用户输入并接收响应
    var botResponse = '<div class="bot-message">This is a response from PassGPT.</div>';
    chatBox.innerHTML += botResponse;

    // 滚动到底部
    chatBox.scrollTop = chatBox.scrollHeight;

    // 清空输入框
    document.getElementById("user-input").value = "";

    // 更新 Markdown 输出
    updateMarkdownOutput();
}



function updateMarkdownOutput() {
    var chatBox = document.getElementById("chat-box");
    var messages = chatBox.getElementsByClassName("user-message");
    var markdownOutput = document.getElementById("markdown-output");
    var markdownText = "";

    for (var i = 0; i < messages.length; i++) {
        markdownText += "- " + messages[i].innerText.trim() + "\n";
    }

    markdownOutput.innerText = markdownText;
}


function formatMarkdown(message) {
    // 将用户输入的文本格式化为 Markdown 格式
    // 这里可以根据需要进行 Markdown 格式的转换，例如加粗、斜体等
    return '**' + message + '**';
}

document.getElementById("user-input").addEventListener("keyup", function(event) {
    if (event.key === "Enter") {
        var userInput = document.getElementById("user-input").value;
        sendMessage(formatMarkdown(userInput));
    }
});

// 接口GPT
function sendMessage2() {
    var userInput = document.getElementById("user-input").value;
    if (userInput.trim() === "") {
        return;
    }

    var chatBox = document.getElementById("chat-box");
    var userMessage = '<div class="user-message">' + userInput + '</div>';
    chatBox.innerHTML += userMessage;

    // 发送用户输入到服务器的 GPT 模型进行处理
    fetch('/your-gpt-endpoint', {
        method: 'POST',
        body: JSON.stringify({ input: userInput }),
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        var botResponse = '<div class="bot-message">' + data.output + '</div>';
        chatBox.innerHTML += botResponse;
        chatBox.scrollTop = chatBox.scrollHeight;
    });

    // 清空输入框
    document.getElementById("user-input").value = "";
}


// 在这个示例中，您需要将 /your-gpt-endpoint 替换为您的 GPT 模型的实际端点。
// 然后，您需要在服务器端设置一个接受 POST 请求的端点，处理用户输入并返回模型的输出。
// 最后，将模型的输出作为 JSON 数据发送回客户端，并将其显示在界面上。