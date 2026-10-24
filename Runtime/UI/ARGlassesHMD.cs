using UnityEngine.InputSystem;
using UnityEngine.InputSystem.Layouts;

namespace ArGlassesSDK.UI
{
    /// <summary>
    /// A virtual tracked device that receives the head pose reported by the AR
    /// glasses. ARPoseProvider writes poses into it each frame, and the Input
    /// System TrackedPoseDriver on the same GameObject drives its Transform
    /// from the inherited devicePosition and deviceRotation controls.
    /// updateBeforeRender opts the device into the Input System's
    /// BeforeRender update so the driver can sample a fresh pose for rendering.
    /// </summary>
    [InputControlLayout(displayName = "AR Glasses HMD", updateBeforeRender = true)]
    public class ARGlassesHMD : TrackedDevice
    {
    }
}
