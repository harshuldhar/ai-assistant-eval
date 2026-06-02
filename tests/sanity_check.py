"""
sanity_check.py — Quick connectivity test for both assistants.
Run this immediately after filling in your .env file to confirm both
APIs are working before building further.

Usage:  python tests/sanity_check.py
"""

import sys
import os

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_frontier():
    print("\n🔵 Testing Frontier (Gemini 2.0 Flash)...")
    try:
        from assistants.frontier_assistant import FrontierAssistant
        a = FrontierAssistant()
        response, latency = a.chat("Say exactly: 'Frontier OK'")
        print(f"   Response: {response}")
        print(f"   Latency:  {latency:.0f}ms")
        assert len(response) > 0, "Empty response"

        # Multi-turn test
        a.chat("My name is TestUser")
        r2, _ = a.chat("What is my name?")
        assert "TestUser" in r2, f"Memory failed: {r2}"
        print("   ✅ Multi-turn memory: PASS")
        print("   ✅ Frontier: PASS")
        return True
    except Exception as e:
        print(f"   ❌ Frontier FAILED: {e}")
        return False

def test_oss():
    print("\n🟢 Testing OSS (Qwen2.5-0.5B-Instruct via HF)...")
    try:
        from assistants.oss_assistant import OSSAssistant
        a = OSSAssistant()
        response, latency = a.chat("Say exactly: 'OSS OK'")
        print(f"   Response: {response[:100]}")
        print(f"   Latency:  {latency:.0f}ms")
        assert len(response) > 0, "Empty response"
        print("   ✅ OSS: PASS")
        return True
    except Exception as e:
        print(f"   ❌ OSS FAILED: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("  AI Assistant Eval — Sanity Check")
    print("=" * 50)

    frontier_ok = test_frontier()
    oss_ok = test_oss()

    print("\n" + "=" * 50)
    print(f"  Frontier: {'✅ PASS' if frontier_ok else '❌ FAIL'}")
    print(f"  OSS:      {'✅ PASS' if oss_ok else '❌ FAIL'}")
    print("=" * 50)

    if frontier_ok and oss_ok:
        print("\n🚀 Both assistants working! You're ready to run the full app.")
        print("   Next: streamlit run ui/app.py")
    else:
        print("\n⚠️  Fix the failing assistant before proceeding.")
        print("   Check your .env file for missing or incorrect API keys.")
        sys.exit(1)
