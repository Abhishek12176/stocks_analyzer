import { apiPost } from "./api";
import type { ChatResponse, ChatTurn } from "@/types/chat";

export async function sendChatMessage(
  message: string,
  history: ChatTurn[]
): Promise<ChatResponse> {
  return apiPost<ChatResponse>("/chat", {
    message,
    history: history.slice(-8),
  });
}
