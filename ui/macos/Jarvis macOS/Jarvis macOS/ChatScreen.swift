import SwiftUI
import Combine
import Foundation


struct ChatScreen: View {
    @State private var username = NSFullUserName()
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
        let prompt = ["actor": "\(username)", "input": "\(message)"]
        messages.append("\(username): \(message)")
        
        let url = URL(string: "http://127.0.0.1:9480/chat")!
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: prompt, options: .prettyPrinted)
        } catch let error {
            print("JSON serialization failed: \(error)")
            return
        }
        
        let task = URLSession.shared.dataTask(with: request) { data, response, error in
            if let data = data {
                do {
                    if let json = try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any] {
                        print("Response JSON: \(json)")
                        if let actor = json["actor"] as? String, let content = json["content"] as? String {
                            let capitalizedActor = actor.prefix(1).uppercased() + actor.dropFirst()
                            messages.append("\(capitalizedActor): \(content)")
                        }
                    }
                } catch let error {
                    print("JSON deserialization failed: \(error)")
                }
            } else if let error = error {
                print("HTTP Request Failed: \(error)")
            }
        }
        
        task.resume()
        
        DispatchQueue.main.async {
            message = ""
        }
    }
}
