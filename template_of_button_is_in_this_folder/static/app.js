class Chatbox {
    constructor() {
        this.args = {
            openButton: document.querySelector('.chatbox__button'),
            chatBox: document.querySelector('.chatbox__support'),
            sendButton: document.querySelector('.send__button')
        }

        this.state = false;
        this.messages = [];
    }

    display() {
        const {openButton, chatBox, sendButton} = this.args;

        openButton.addEventListener('click', () => this.toggleState(chatBox))

        sendButton.addEventListener('click', () => this.onSendButton(chatBox))

        const node = chatBox.querySelector('input');
        node.addEventListener("keyup", ({key}) => {
            if (key === "Enter") {
                this.onSendButton(chatBox)
            }
        })
    }

    toggleState(chatbox) {
        this.state = !this.state;

        // show or hides the box
        if(this.state) {
            chatbox.classList.add('chatbox--active')
        } else {
            chatbox.classList.remove('chatbox--active')
        }
    }

    clear_button(chatbox) {
            var lv1 = document.getElementById("lv1");
            var lv2 = document.getElementById("lv2");
            var lv3 = document.getElementById("lv3");

            lv1.style.display = "none";
            lv2.style.display = "none";
            lv3.style.display = "none";
    }


    onSendButton(chatbox) {
        var textField = chatbox.querySelector('input');
        let text1 = textField.value
        if (text1 === "") {
            return;
        }

        let msg1 = { name: "User", message: text1 }
        this.messages.push(msg1);

        if (text1.includes("level")){
            var lv1 = document.getElementById("lv1");
            var lv2 = document.getElementById("lv2");
            var lv3 = document.getElementById("lv3");

            console.log("Hello");
            lv1.style.display = "flex";
            lv2.style.display = "flex";
            lv3.style.display = "flex";


            lv1.onclick = function() {

                let lv1_message = { name: "Sam", message: "lv1" };

                this.messages.push(lv1_message);
                this.updateChatText(chatbox);
                textField.value = '';

                lv1.style.display = "none";
                lv2.style.display = "none";
                lv3.style.display = "none";

            }.bind(this, chatbox);

            lv2.onclick = function() {
//                document.getElementById("msg").innerText = "Hello lv2";
                let lv2_message = { name: "Sam", message: "lv2" };

                this.messages.push(lv2_message);
                this.updateChatText(chatbox);
                textField.value = '';

                lv1.style.display = "none";
                lv2.style.display = "none";
                lv3.style.display = "none";

            }.bind(this, chatbox);

            lv3.onclick = function() {
//                document.getElementById("msg").innerText = "Hello lv3";
                let lv3_message = { name: "Sam", message: "lv3" };

                this.messages.push(lv3_message);
                this.updateChatText(chatbox);
                textField.value = '';

                lv1.style.display = "none";
                lv2.style.display = "none";
                lv3.style.display = "none";
            }.bind(this, chatbox);

        } else if (text1.includes("ปัญหา")){
            var lv1 = document.getElementById("lv1");
            var lv2 = document.getElementById("lv2");
            var lv3 = document.getElementById("lv3");

            lv1.style.display = "flex";
            lv2.style.display = "flex";
            lv3.style.display = "flex";


            lv1.onclick = function() {
                let answer = "answer problem 1"
                let lv1_message = { name: "Sam", message: answer };

                this.messages.push(lv1_message);
                this.updateChatText(chatbox);
                textField.value = '';

                lv1.style.display = "none";
                lv2.style.display = "none";
                lv3.style.display = "none";

            }.bind(this, chatbox);

            lv2.onclick = function() {
                let answer = "answer problem 2"
                let lv2_message = { name: "Sam", message: "answer problem 2" };

                this.messages.push(lv2_message);
                this.updateChatText(chatbox);
                textField.value = '';

                lv1.style.display = "none";
                lv2.style.display = "none";
                lv3.style.display = "none";

            }.bind(this, chatbox);

            lv3.onclick = function() {
                let answer = "answer problem 3"
                let lv3_message = { name: "Sam", message: answer };

                this.messages.push(lv3_message);
                this.updateChatText(chatbox);
                textField.value = '';

                lv1.style.display = "none";
                lv2.style.display = "none";
                lv3.style.display = "none";
            }.bind(this, chatbox);

        } else {

            fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            body: JSON.stringify({ message: text1 }),
            mode: 'cors',
            headers: {
              'Content-Type': 'application/json'
            },
          })
          .then(r => r.json())
          .then(r => {
            let msg2 = { name: "Sam", message: r.answer };
            this.messages.push(msg2);
//            console.log(msg2);
            this.updateChatText(chatbox)
            textField.value = ''

        }).catch((error) => {
            console.error('Error:', error);
            this.updateChatText(chatbox)
            textField.value = ''
          });

        }
    }

    updateChatText(chatbox) {
        var html = '';
        this.messages.slice().reverse().forEach(function(item, index) {
            if (item.name === "Sam")
            {
                html += '<div class="messages__item messages__item--visitor">' + item.message + '</div>'
            }
            else
            {
                html += '<div class="messages__item messages__item--operator">' + item.message + '</div>'
            }
          });

        const chatmessage = chatbox.querySelector('.chatbox__messages');
        chatmessage.innerHTML = html;
    }
}


const chatbox = new Chatbox();
chatbox.display();