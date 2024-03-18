//
//  ContentView.swift
//  Jarvis macOS
//
//  Created by Bradley Reimers on 3/17/24.
//

import SwiftUI

struct ContentView: View {

    @State private var username: String = ""
    var body: some View {
        ChatScreen()
    }
}

#Preview {
    ContentView()
}

