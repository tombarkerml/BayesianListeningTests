var audioCtx = new (window.AudioContext || window.webkitAudioContext)();

    // create Oscillator node
    var osc1 = audioCtx.createOscillator();
    var osc2 = audioCtx.createOscillator();
    var osc3 = audioCtx.createOscillator();
    var osc4 = audioCtx.createOscillator();
    var osc5 = audioCtx.createOscillator();


    var gainNode1 = audioCtx.createGain();
    var gainNode2 = audioCtx.createGain();
    var gainNode3 = audioCtx.createGain();
    var gainNode4 = audioCtx.createGain();
    var gainNode5 = audioCtx.createGain();

    //osc1.type = 'sine';
    //osc1.frequency.value=440; // value in hertz
    //osc1.connect(gainNode1);
    //osc1.start();

    //osc1.connect(gainNode1);
    //gainNode1.connect(audioCtx.destination);

    //gainNode1.gain.value = 1;


    function connectOscillator(oscillator, gainNode){
        oscillator.type = 'sine'
        oscillator.frequency.value=440;
        oscillator.connect(gainNode)
        oscillator.start();
        gainNode.gain.value=1;
        gainNode.connect(audioCtx.destination)
    }

    connectOscillator(osc1, gainNode1)
    connectOscillator(osc2, gainNode2)
    connectOscillator(osc3, gainNode3)
    connectOscillator(osc4, gainNode4)
    connectOscillator(osc5, gainNode5)