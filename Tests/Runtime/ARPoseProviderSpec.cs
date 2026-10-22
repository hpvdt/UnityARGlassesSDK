using ArGlassesSDK.UI;
using NUnit.Framework;
using UnityEngine;

namespace ArGlassesSDK.Tests
{
    [TestFixture]
    public class ARPoseProviderSpec
    {
        // Exposes the protected Attitude state for assertions;
        // zeroing never touches the native driver so it is safe to test
        private class TestablePoseProvider : ARPoseProvider
        {
            public Rotation TestAttitude => Attitude;
        }

        private TestablePoseProvider provider;

        [SetUp]
        public void SetUp()
        {
            provider = new GameObject("ARPoseProviderSpec").AddComponent<TestablePoseProvider>();
        }

        [TearDown]
        public void TearDown()
        {
            if (provider != null) Object.DestroyImmediate(provider.gameObject);
        }

        [Test]
        public void ZeroY_CompensatesGlassesYaw()
        {
            provider.TestAttitude.Glasses = Quaternion.Euler(0f, 90f, 0f);

            provider.ZeroY();

            AssertLessThanOneTenthDegree(Quaternion.Euler(0f, -90f, 0f), provider.TestAttitude.Zeroing);
        }

        [Test]
        public void ZeroY_CompensatesMouseYaw()
        {
            provider.TestAttitude.Mouse = Quaternion.Euler(0f, 45f, 0f);

            provider.ZeroY();

            AssertLessThanOneTenthDegree(Quaternion.Euler(0f, -45f, 0f), provider.TestAttitude.Zeroing);
        }

        [Test]
        public void ZeroY_LeavesPitchAndRollAlone()
        {
            provider.TestAttitude.Glasses = Quaternion.Euler(30f, 90f, 15f);

            provider.ZeroY();

            var zeroingEuler = provider.TestAttitude.Zeroing.eulerAngles;
            Assert.AreEqual(0f, zeroingEuler.x, 0.001f);
            Assert.AreEqual(0f, zeroingEuler.z, 0.001f);
        }

        [Test]
        public void ZeroY_CancelsYawOfCompoundRotation()
        {
            provider.TestAttitude.Glasses = Quaternion.Euler(0f, 30f, 0f);
            provider.TestAttitude.Mouse = Quaternion.Euler(0f, 60f, 0f);

            provider.ZeroY();

            var compound = provider.TestAttitude.Mouse * provider.TestAttitude.Zeroing *
                           provider.TestAttitude.Glasses;
            Assert.AreEqual(0f, compound.eulerAngles.y, 0.001f);
        }

        [Test]
        public void ZeroXY_CompensatesGlassesPitchAndYaw()
        {
            provider.TestAttitude.Glasses = Quaternion.Euler(30f, 90f, 0f);

            provider.ZeroXY();

            AssertLessThanOneTenthDegree(Quaternion.Euler(-30f, -90f, 0f), provider.TestAttitude.Zeroing);
        }

        [Test]
        public void ZeroXY_LeavesRollAlone()
        {
            provider.TestAttitude.Glasses = Quaternion.Euler(30f, 90f, 15f);

            provider.ZeroXY();

            var zeroingEuler = provider.TestAttitude.Zeroing.eulerAngles;
            Assert.AreEqual(0f, zeroingEuler.z, 0.001f);
        }

        [Test]
        public void ZeroXY_CancelsPitchOfCompoundRotation()
        {
            provider.TestAttitude.Glasses = Quaternion.Euler(20f, 0f, 0f);
            provider.TestAttitude.Mouse = Quaternion.Euler(-10f, 0f, 0f);

            provider.ZeroXY();

            var compound = provider.TestAttitude.Mouse * provider.TestAttitude.Zeroing *
                           provider.TestAttitude.Glasses;
            Assert.AreEqual(0f, compound.eulerAngles.x, 0.001f);
        }

        [Test]
        public void ZeroXY_CancelsYawOfCompoundRotation()
        {
            provider.TestAttitude.Glasses = Quaternion.Euler(0f, 20f, 0f);
            provider.TestAttitude.Mouse = Quaternion.Euler(0f, 35f, 0f);

            provider.ZeroXY();

            var compound = provider.TestAttitude.Mouse * provider.TestAttitude.Zeroing *
                           provider.TestAttitude.Glasses;
            Assert.AreEqual(0f, compound.eulerAngles.y, 0.001f);
        }

        [Test]
        public void ZeroXY_ReplacesPreviousZeroing()
        {
            provider.TestAttitude.Glasses = Quaternion.Euler(0f, 90f, 0f);
            provider.ZeroXY();
            var firstZeroing = provider.TestAttitude.Zeroing;

            provider.TestAttitude.Glasses = Quaternion.Euler(0f, 30f, 0f);
            provider.TestAttitude.Mouse = Quaternion.identity;
            provider.ZeroXY();

            // the second call replaces the zeroing with the one cancelling
            // the current attitude, it does not stack onto the first one
            Assert.AreNotEqual(firstZeroing, provider.TestAttitude.Zeroing);
            AssertLessThanOneTenthDegree(Quaternion.Euler(0f, -30f, 0f), provider.TestAttitude.Zeroing);
        }

        // NUnit compares quaternions with bit-exact equality, but the euler
        // decomposition feeding the zeroing is subject to float rounding,
        // so the assertions must allow a small angular tolerance
        private static void AssertLessThanOneTenthDegree(Quaternion expected, Quaternion actual)
        {
            Assert.AreEqual(0f, Quaternion.Angle(expected, actual), 0.1f);
        }
    }
}
