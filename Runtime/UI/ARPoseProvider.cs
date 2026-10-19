using System;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Experimental.XR.Interaction;
using UnityEngine.SpatialTracking;
using Quaternion = UnityEngine.Quaternion;
using Vector3 = UnityEngine.Vector3;

namespace ArGlassesSDK.UI
{
    public class ARPoseProvider : BasePoseProvider
    {
        public enum ReadMethods
        {
            Quaternion,
            EulerAngle,
            Debugging
        }

        public ReadMethods readMethod = ReadMethods.EulerAngle;

        public bool verboseLogging = false;

        public float mouseSensitivity = 100.0f;

        // 3-axis position is mapped from the linear velocity estimation with
        // this ratio: meters of camera offset per m/s of estimated head velocity
        public float velocityToPositionRatio = 1.0f;

#if (UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN)
        [DllImport("ar_drivers.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int StartConnection(out IntPtr error);

        [DllImport("ar_drivers.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int StopConnection(out IntPtr error);

        [DllImport("ar_drivers.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr GetQuaternion(out IntPtr error);

        [DllImport("ar_drivers.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr GetEuler(out IntPtr error);

        [DllImport("ar_drivers.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr GetLinearVelocity(out IntPtr error);

#elif (UNITY_EDITOR_OSX || UNITY_STANDALONE_OSX)
        [DllImport("libar_drivers.dylib", CallingConvention = CallingConvention.Cdecl)]
        public static extern int StartConnection(out IntPtr error);

        [DllImport("libar_drivers.dylib", CallingConvention = CallingConvention.Cdecl)]
        public static extern int StopConnection(out IntPtr error);

        [DllImport("libar_drivers.dylib", CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr GetEuler(out IntPtr error);

        [DllImport("libar_drivers.dylib", CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr GetQuaternion(out IntPtr error);

        [DllImport("libar_drivers.dylib", CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr GetLinearVelocity(out IntPtr error);

#else
        [DllImport("libar_drivers.so", CallingConvention = CallingConvention.Cdecl)]
        public static extern int StartConnection(out IntPtr error);

        [DllImport("libar_drivers.so", CallingConvention = CallingConvention.Cdecl)]
        public static extern int StopConnection(out IntPtr error);

        [DllImport("libar_drivers.so", CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr GetEuler(out IntPtr error);

        [DllImport("libar_drivers.so", CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr GetQuaternion(out IntPtr error);

        [DllImport("libar_drivers.so", CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr GetLinearVelocity(out IntPtr error);

#endif

        // The native error pointer is valid only until the next native call,
        // so it is copied into a managed string immediately after each call;
        // null when the native operation succeeded.
        private static string ReadNativeError(IntPtr error)
        {
            return error == IntPtr.Zero ? null : Marshal.PtrToStringAnsi(error);
        }

        // The per-frame getters keep the last pose on failure, so their error
        // messages are only logged when verbose logging is turned on. The
        // native message already names the failed operation and its cause.
        protected void LogNativeError(IntPtr error)
        {
            if (!verboseLogging) return;
            var message = ReadNativeError(error);
            if (message != null) Debug.LogWarning(message);
        }

        protected enum ConnectionStates
        {
            Disconnected = 0,
            Offline = 1,
            StandBy = 2,

            Connected = 3
        }

        protected static ConnectionStates ConnectionState = ConnectionStates.Disconnected;

        public bool IsConnecting()
        {
            return ConnectionState is ConnectionStates.Connected or ConnectionStates.StandBy;
        }

        public void TryConnect()
        {
            // Start the connection
            var code = StartConnection(out var error);
            if (code == 1)
            {
                ConnectionState = ConnectionStates.StandBy;
                Debug.Log("Glasses standing by");
            }
            else
            {
                var message = ReadNativeError(error);
                Debug.LogError("Connection error: return code " + code + (message == null ? "" : ": " + message));
            }
        }


        public void TryDisconnect()
        {
            if (IsConnecting())
            {
                var code = StopConnection(out var error);
                if (code == 1)
                {
                    ConnectionState = ConnectionStates.Disconnected;
                    Debug.Log("Glassed disconnected");
                }
                else
                {
                    ConnectionState = ConnectionStates.Offline;
                    var message = ReadNativeError(error);
                    Debug.LogWarning("Glassed disconnected with error: return code " + code +
                                      (message == null ? "" : ": " + message));
                }
            }
            else
            {
                Debug.Log("Glassed not connected, no need to disconnect");
            }
        }

        // these are invoked by UI events, which can only bind
        // methods on the provider component itself

        public void ZeroY()
        {
            Attitude.ZeroY();
        }

        public void ZeroXY()
        {
            Attitude.ZeroXY();
        }

        private static readonly Quaternion Q_ID = Quaternion.identity.normalized;

        public class Rotation
        {
            protected readonly ARPoseProvider Outer;

            public Rotation(ARPoseProvider outer)
            {
                Outer = outer;
            }

            public Quaternion Glasses = Quaternion.identity;
            public Quaternion Mouse = Quaternion.identity;
            public Quaternion Zeroing = Quaternion.identity;

            private Vector3 _mouseEuler = Vector3.zero;

            private Vector3 _zeroingEuler = Vector3.zero;

            private float[] GetEulerArray()
            {
                var ptr = GetEuler(out var error);
                Outer.LogNativeError(error);
                var r = new float[3];
                Marshal.Copy(ptr, r, 0, 3);
                return r;
            }

            private static readonly Quaternion Q_Neutral = Quaternion.Euler(90f, 0f, 0f).normalized;


            // the following rules are chosen to be
            //  compatible with alternative Windows driver (https://github.com/wheaney/OpenVR-xrealAirGlassesHMD)

            protected Quaternion Read_direct()
            {
                var ptr = GetQuaternion(out var error);
                Outer.LogNativeError(error);
                // receiving in WIJK order (left hand)
                // see https://github.com/xioTechnologies/Fusion/blob/e7d2b41e6506fa9c85492b91becf003262f14977/Fusion/FusionMath.h#L36

                var arr = new float[4];
                Marshal.Copy(ptr, arr, 0, 4);

                var qRaw = new Quaternion(-arr[1], -arr[3], -arr[2], arr[0]);

                if (Outer.verboseLogging) Debug.Log($"Quaternion raw: {qRaw.x}, {qRaw.y}, {qRaw.z}, {qRaw.w}");

                // converting to IKJW order (right hand)
                // sequence (1, -3, -2, 0) is for chiral conversion
                // see https://stackoverflow.com/questions/28673777/convert-quaternion-from-right-handed-to-left-handed-coordinate-system
                // neutral position is 90 degree pitch downward

                var qNormalised = qRaw.normalized;
                // some driver may have all 0 reading, which will be converted to identity quaternion

                var q = qNormalised * Q_Neutral;
                // var q = qRaw * QNeutral;
                // Debug.Log($"Quaternion after: {q.x}, {q.y}, {q.z}, {q.w}");
                return q;
            }

            protected Quaternion Read_euler()
            {
                var arr = GetEulerArray();
                // receiving in FRU order (left hand axes, right hand rotation)
                // Forward - roll
                // Right - pitch
                // Up - yaw

                var roll = arr[0];
                var pitch = arr[1];
                var yaw = arr[2];

                var arr2 = new Vector3(-(pitch - 90f), -yaw, -roll);
                if (Outer.verboseLogging) Debug.Log($"Euler raw: {arr2[0]}, {arr2[1]}, {arr2[2]}");

                // converting to LDB order (right hand axes, right hand rotation)
                // Left - pitch
                // Down - yaw
                // Backward - roll
                // neutral position is 90 degree pitch downward

                var q = Quaternion.Euler(arr2[0], arr2[1], arr2[2]);
                return q;
            }


            private static Vector3 ClampTo180(Vector3 v)
            {
                return new Vector3(
                    ClampAngle180(v.x),
                    ClampAngle180(v.y),
                    ClampAngle180(v.z)
                );
            }

            private static float ClampAngle180(float angle)
            {
                angle = angle % 360;
                if (angle > 180)
                    angle -= 360;
                else if (angle < -180)
                    angle += 360;
                return angle;
            }

            protected Quaternion Read_debugging()
            {
                var r1 = Read_direct();
                var r2 = Read_euler();

                var errorBound = 10f;
                Debug.Assert(r1 == r1.normalized, "unnormalised quaternion");
                Debug.Assert(r2 == r2.normalized, "unnormalised quaternion from Euler angle");

                {
                    Vector3 fwd;
                    {
                        var q = Quaternion.Inverse(r1) * r2;
                        Debug.Assert(r1 * q == r2, "fwd error!");
                        fwd = ClampTo180(q.eulerAngles);
                    }

                    Vector3 rev;
                    {
                        var q = Quaternion.Inverse(r2) * r1;
                        Debug.Assert(r2 * q == r1, "rev error!");
                        rev = ClampTo180(q.eulerAngles);
                    }

                    var angle = Quaternion.Angle(r1, r2);

                    var errorInfo =
                        $"inconsistency between Read_direct and Read_euler:\n\tfwd = {fwd} ; rev = {rev} ; angle = {angle}";

                    Debug.Assert(fwd.magnitude <= errorBound, errorInfo);
                }

                return r2;
            }

            protected Quaternion Read()
            {
                switch (Outer.readMethod)
                {
                    case ReadMethods.Quaternion:
                        return Read_direct();
                    case ReadMethods.Debugging:
                        return Read_debugging();
                    default:
                        return Read_euler();
                }
            }

            public void UpdateFromGlasses()
            {
                var reading = Read();

                var effective = reading;
                // var effective = (reading * Q_NEUTRAL).normalized;

                if (ConnectionState == ConnectionStates.StandBy)
                {
                    var fromNeutral = Quaternion.Angle(reading, Q_Neutral);
                    if (!(reading.Equals(Q_Neutral) || fromNeutral < 1.0))
                    {
                        ConnectionState = ConnectionStates.Connected;
                        Debug.Log("Glasses connected, start reading");
                        Glasses = effective;
                    }
                    else
                    {
                        if (Outer.verboseLogging) Debug.Log("Glasses has no reading");
                    }
                }
                else
                {
                    Glasses = effective;
                }
            }

            public void UpdateFromMouse()
            {
                var deltaY = Input.GetAxis("Mouse X") * Outer.mouseSensitivity * Time.deltaTime;
                var deltaX = -Input.GetAxis("Mouse Y") * Outer.mouseSensitivity * Time.deltaTime;
                // Mouse & Unity XY axis are opposite

                _mouseEuler += new Vector3(deltaX, deltaY, 0.0f);

                // Debug.Log("mouse pressed:" + FromMouseXY);

                Mouse = Quaternion.Euler(-_mouseEuler);
            }

            public void ZeroY()
            {
                var fromGlassesY = (Glasses * Mouse).eulerAngles.y;
                _zeroingEuler.y = -fromGlassesY;
                Zeroing = Quaternion.Euler(_zeroingEuler);
            }

            public void ZeroXY()
            {
                var fromGlasses = (Glasses * Mouse).eulerAngles;
                _zeroingEuler.x = -fromGlasses.x;
                _zeroingEuler.y = -fromGlasses.y;
                Zeroing = Quaternion.Euler(_zeroingEuler);
            }
        }

        private Rotation _attitude;

        protected Rotation Attitude => _attitude ??= new Rotation(this);

        private Translation _position;

        protected Translation Position => _position ??= new Translation(this);

        public class Translation
        {
            protected readonly ARPoseProvider Outer;

            public Translation(ARPoseProvider outer)
            {
                Outer = outer;
            }

            // velocity from the glasses, in Unity world axes, m/s
            private Vector3 _fromGlasses = Vector3.zero;

            // position offset in Unity world axes, meters; the estimator's
            // zero-velocity regression recenters it whenever the glasses rest
            public Vector3 Offset => _fromGlasses * Outer.velocityToPositionRatio;

            private float[] GetVelocityArray()
            {
                var ptr = GetLinearVelocity(out var error);
                Outer.LogNativeError(error);
                var r = new float[3];
                Marshal.Copy(ptr, r, 0, 3);
                return r;
            }

            public void UpdateFromGlasses()
            {
                var arr = GetVelocityArray();
                // receiving in FRU order (left hand axes), the same presentation as GetEuler
                // Forward, Right, Up

                // converting to RUF order (Unity: x=Right, y=Up, z=Forward)
                var mapped = new Vector3(arr[1], arr[2], arr[0]);

                // the velocity is estimated in the glasses' FRD world frame, the Zeroing
                // quaternion is that frame's yaw offset into Unity world (the world-frame
                // part of the attitude's Mouse * Zeroing * Glasses conversion chain), so
                // translation stays aligned with the zeroed view; Zeroing is yaw-only
                _fromGlasses = Outer.Attitude.Zeroing * mapped;

                if (Outer.verboseLogging) Debug.Log($"Velocity mapped: {mapped}");
            }
        }

        // Start is called before the first frame update
        private void Start()
        {
            TryConnect();
        }

        private void OnDestroy()
        {
            TryDisconnect();
        }

        // Update Pose
        public override PoseDataFlags GetPoseFromProvider(out Pose output)
        {
            if (IsConnecting())
            {
                Attitude.UpdateFromGlasses();
                Position.UpdateFromGlasses();
            }

            if (Input.GetMouseButton(1)) Attitude.UpdateFromMouse();

            var compound = Attitude.Mouse * Attitude.Zeroing * Attitude.Glasses;

            output = new Pose(Position.Offset, compound);
            return PoseDataFlags.Rotation | PoseDataFlags.Position;
        }
    }
}