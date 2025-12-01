"""
Test AudioSocket Protocol
Verifies encoding/decoding works correctly.
"""

import sys

try:
    from audio_socket_protocol import AudioSocketProtocol, MessageType
except ImportError as e:
    print(f"❌ IMPORT ERROR: {e}")
    print("Make sure you're in the correct directory with audio_socket_protocol.py")
    sys.exit(1)


def test_uuid_message():
    """Test UUID message encoding"""
    print("Testing UUID message encoding...", end=" ")

    uuid_msg = AudioSocketProtocol.create_uuid_message()
    assert len(uuid_msg) == 19, f"UUID message should be 19 bytes, got {len(uuid_msg)}"

    # Verify header
    assert uuid_msg[0] == MessageType.UUID, f"First byte should be {MessageType.UUID}, got {uuid_msg[0]}"
    assert uuid_msg[1:3] == b'\x00\x10', f"Length should be 16 (big-endian \\x00\\x10), got {uuid_msg[1:3].hex()}"

    print("✅ PASSED")


def test_audio_message():
    """Test audio message encoding"""
    print("Testing audio message encoding...", end=" ")

    pcm_data = b'\x00' * 320
    audio_msg = AudioSocketProtocol.create_audio_message(pcm_data)
    assert len(audio_msg) == 323, f"Audio message should be 323 bytes, got {len(audio_msg)}"

    # Verify header
    assert audio_msg[0] == MessageType.AUDIO, f"First byte should be {MessageType.AUDIO}, got {audio_msg[0]}"
    assert audio_msg[1:3] == b'\x01\x40', f"Length should be 320 (0x0140 big-endian), got {audio_msg[1:3].hex()}"

    print("✅ PASSED")


def test_terminate_message():
    """Test terminate message encoding"""
    print("Testing terminate message encoding...", end=" ")

    term_msg = AudioSocketProtocol.create_terminate_message()
    assert len(term_msg) == 3, f"Terminate message should be 3 bytes, got {len(term_msg)}"

    # Verify header
    assert term_msg[0] == MessageType.TERMINATE, f"First byte should be {MessageType.TERMINATE}, got {term_msg[0]}"
    assert term_msg[1:3] == b'\x00\x00', f"Length should be 0, got {term_msg[1:3].hex()}"

    print("✅ PASSED")


def test_frame_encoding():
    """Test general frame encoding"""
    print("Testing frame encoding...", end=" ")

    payload = b'test_payload'
    frame = AudioSocketProtocol.encode_frame(MessageType.AUDIO, payload)

    assert frame[0] == MessageType.AUDIO, f"Type should be {MessageType.AUDIO}, got {frame[0]}"
    assert frame[1:3] == b'\x00\x0c', f"Length should be 12 (0x000c), got {frame[1:3].hex()}"
    assert frame[3:] == payload, f"Payload mismatch"

    print("✅ PASSED")


def main():
    """Run all tests"""
    print("=" * 60)
    print("AudioSocket Protocol Tests")
    print("=" * 60)
    print()

    failed = False

    try:
        test_uuid_message()
    except AssertionError as e:
        print(f"❌ FAILED: {e}")
        failed = True
    except Exception as e:
        print(f"❌ ERROR: {e}")
        failed = True

    try:
        test_audio_message()
    except AssertionError as e:
        print(f"❌ FAILED: {e}")
        failed = True
    except Exception as e:
        print(f"❌ ERROR: {e}")
        failed = True

    try:
        test_terminate_message()
    except AssertionError as e:
        print(f"❌ FAILED: {e}")
        failed = True
    except Exception as e:
        print(f"❌ ERROR: {e}")
        failed = True

    try:
        test_frame_encoding()
    except AssertionError as e:
        print(f"❌ FAILED: {e}")
        failed = True
    except Exception as e:
        print(f"❌ ERROR: {e}")
        failed = True

    print()
    if failed:
        print("❌ Some tests FAILED")
        sys.exit(1)
    else:
        print("🎉 All protocol tests PASSED!")
        sys.exit(0)


if __name__ == "__main__":
    main()
