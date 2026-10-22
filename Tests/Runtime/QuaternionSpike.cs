using System.Collections;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace ArGlassesSDK.Tests
{
    [Ignore("spike")]
    public class QuaternionSpike
    {
        private static List<Quaternion> Anchors()
        {
            // not used, only for debugging

            var anchors = new List<Quaternion>();
            anchors.Add(Quaternion.Euler(90f, 0f, 0f).normalized);
            anchors.Add(Quaternion.Euler(0f, 90f, 0f).normalized);
            anchors.Add(Quaternion.Euler(0f, 0f, 90f).normalized);
            return anchors;
        }

        // A Test behaves as an ordinary method
        [Test]
        public void PrintAnchors()
        {
            var aa = Anchors();
            foreach (var a in aa) Debug.Log(a);
            foreach (var a in aa)
            {
                var inv = Quaternion.Inverse(a);
                Debug.Log(inv);
            }
            // Use the Assert class to test conditions
        }

        // A UnityTest behaves like a coroutine in Play Mode. In Edit Mode you can use
        // `yield return null;` to skip a frame.
        [UnityTest]
        public IEnumerator SanityWithEnumeratorPasses()
        {
            // Use the Assert class to test conditions.
            // Use yield to skip a frame.
            yield return null;
        }
    }
}