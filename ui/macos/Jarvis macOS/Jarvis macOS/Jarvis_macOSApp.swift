//
//  Jarvis_macOSApp.swift
//  Jarvis macOS
//
//  Created by Bradley Reimers on 3/17/24.
//

import SwiftUI

@main
struct Jarvis_macOSApp: App {
    init() {
      startLLMServer()
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
    
    func startLLMServer() {
        guard let binaryURL = Bundle.main.url(forResource: "api", withExtension: nil) else {
            print("Error: Could not find server binary in the app bundle.")
            return
        }

        let process = Process()
        process.executableURL = binaryURL
        process.launch()
    }
}
