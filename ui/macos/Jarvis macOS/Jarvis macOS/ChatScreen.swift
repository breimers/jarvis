import SwiftUI
import Combine
import Foundation


struct ChatScreen: View {
    @State private var message = ""
    @State private var messages: [String] = []

    var body: some View {
        VStack {
            // Chat history.
            List(messages, id: \.self) { message in
                Text(message)
                    .padding(8)
                    .background(Color.secondary.opacity(0.2))
                    .cornerRadius(5)
                    .padding(5)
            }

            // Message field.
            HStack {
                TextField("Message", text: $message, onCommit: sendMessage)
                    .padding(10)
                    .background(Color.secondary.opacity(0.2))
                    .cornerRadius(5)
                Button(action: sendMessage) {
                    Image(systemName: "arrowshape.turn.up.right")
                        .font(.system(size: 20))
                }
                .padding()
                .disabled(message.isEmpty)
            }
            .padding()
        }
    }

    func sendMessage() {
        messages.append("User: \(message)")
        DispatchQueue.main.async {
            message = "" 
        }
    }
}
