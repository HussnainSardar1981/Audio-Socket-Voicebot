"""
Test AudioSocket Protocol
Verifies encoding/decoding works correctly.
"""

from audio_socket_protocol import AudioSocketProtocol, MessageType


def test_uuid_message():
    """Test UUID message encoding"""
    uuid_msg = AudioSocketProtocol.create_uuid_message()
    assert len(uuid_msg) == 19, f"UUID message should be 19 bytes, got {len(uuid_msg)}"

    # Verify header
    assert uuid_msg[0] == MessageType.UUID, "First byte should be UUID type"
    assert uuid_msg[1:3] == b'\x00\x10', "Length should be 16 (big-endian)"

    print("✅ UUID message test passed")


def test_audio_message():
    """Test audio message encoding"""
    pcm_data = b'\x00' * 320
    audio_msg = AudioSocketProtocol.create_audio_message(pcm_data)
    assert len(audio_msg) == 323, f"Audio message should be 323 bytes, got {len(audio_msg)}"

    # Verify header
    assert audio_msg[0] == MessageType.AUDIO, "First byte should be AUDIO type"
    assert audio_msg[1:3] == b'\x01\x40', "Length should be 320 (0x0140 big-endian)"

    print("✅ Audio message test passed")


def test_terminate_message():
    """Test terminate message encoding"""
    term_msg = AudioSocketProtocol.create_terminate_message()
    assert len(term_msg) == 3, f"Terminate message should be 3 bytes, got {len(term_msg)}"

    # Verify header
    assert term_msg[0] == MessageType.TERMINATE, "First byte should be TERMINATE type"
    assert term_msg[1:3] == b'\x00\x00', "Length should be 0"

    print("✅ Terminate message test passed")


def test_frame_encoding():
    """Test general frame encoding"""
    payload = b'test_payload'
    frame = AudioSocketProtocol.encode_frame(MessageType.AUDIO, payload)

    assert frame[0] == MessageType.AUDIO
    assert frame[1:3] == b'\x00\x0c', f"Length should be 12 (0x000c), got {frame[1:3].hex()}"
    assert frame[3:] == payload

    print("✅ Frame encoding test passed")


if __name__ == "__main__":
    print("Testing AudioSocket Protocol...")
    print()

    test_uuid_message()
    test_audio_message()
    test_terminate_message()
    test_frame_encoding()

    print()
    print("🎉 All protocol tests passed!")
