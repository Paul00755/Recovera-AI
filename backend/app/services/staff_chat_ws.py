import json
import logging
from typing import Dict, Set, Any, Optional
from fastapi import WebSocket

logger = logging.getLogger("recovera.staff_chat_ws")

class StaffChatConnectionManager:
    """Manages real-time WebSockets for internal hospital staff chat, presence & WebRTC calling."""

    def __init__(self):
        # user_id -> Set[WebSocket]
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # user_id -> presence status ("online", "away", "busy", "offline")
        self.user_presence: Dict[str, str] = {}

    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = set()
        self.active_connections[user_id].add(websocket)
        self.user_presence[user_id] = "online"
        logger.info(f"Staff WS connected: user={user_id} (total sockets={len(self.active_connections[user_id])})")
        # Broadcast online status to all staff
        await self.broadcast_presence(user_id, "online")

    def disconnect(self, websocket: WebSocket, user_id: str):
        if user_id in self.active_connections:
            self.active_connections[user_id].discard(websocket)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
                self.user_presence[user_id] = "offline"
                logger.info(f"Staff WS disconnected: user={user_id} (all sockets closed)")
                return True  # user became completely offline
        return False

    def is_user_online(self, user_id: str) -> bool:
        return user_id in self.active_connections and len(self.active_connections[user_id]) > 0

    def get_online_user_ids(self) -> Set[str]:
        return {uid for uid, socks in self.active_connections.items() if socks}

    async def send_to_user(self, user_id: str, message: Dict[str, Any]):
        """Send a JSON event to all active sockets of a specific staff user."""
        if user_id in self.active_connections:
            payload = json.dumps(message)
            dead_sockets = set()
            for sock in list(self.active_connections[user_id]):
                try:
                    await sock.send_text(payload)
                except Exception as e:
                    logger.warning(f"Failed to send WS message to user {user_id}: {e}")
                    dead_sockets.add(sock)
            for dead in dead_sockets:
                self.active_connections[user_id].discard(dead)

    async def broadcast_to_users(self, user_ids: list[str], message: Dict[str, Any]):
        """Send an event to a list of staff user IDs."""
        for uid in user_ids:
            await self.send_to_user(uid, message)

    async def broadcast_all(self, message: Dict[str, Any]):
        """Broadcast an event to all currently connected staff users."""
        payload = json.dumps(message)
        for uid, socks in list(self.active_connections.items()):
            for sock in list(socks):
                try:
                    await sock.send_text(payload)
                except Exception:
                    pass

    async def broadcast_presence(self, user_id: str, status: str, custom_status: Optional[str] = None):
        self.user_presence[user_id] = status
        await self.broadcast_all({
            "type": "presence_update",
            "data": {
                "user_id": user_id,
                "status": status,
                "custom_status": custom_status
            }
        })


staff_ws_manager = StaffChatConnectionManager()
