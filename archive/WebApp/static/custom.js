// function to adjust the sudoku grid
function adjustSudokuBox() {
    $('.sudoku-box').each(function() {
            $(this).css('font-size', Math.max(0.3*parseFloat($(this).parent().css('width'))/9, 10));
            $(this).css('height', parseFloat($(this).parent().css('width'))/9);
            $(this).css('width', parseFloat($(this).parent().css('width'))/9)
        }
    )
}

function fillPuzzle(puzzle) {
    //$('.sudoku').data('puzzle', JSON.stringify({'puzzle': puzzle}));
    $('.sudoku').data('puzzle', puzzle);
    if ($('.sudoku')[0].hasAttribute('data-solution')) {
        $('.sudoku').removeData('solution')
    }
    for (var i = 0; i < puzzle.length; i++) {
        for (var j = 0; j < puzzle[i].length; j++){
            $('#'.concat(i.toString(), '-', j.toString())).removeClass('error');
            if (puzzle[i][j] != 0){
                $('#'.concat(i.toString(), '-', j.toString())).html(puzzle[i][j].toString());
                $('#'.concat(i.toString(), '-', j.toString())).attr('contenteditable', 'false');
                $('#'.concat(i.toString(), '-', j.toString())).addClass('locked')
            }else{
                $('#'.concat(i.toString(), '-', j.toString())).html('');
                $('#'.concat(i.toString(), '-', j.toString())).attr('contenteditable', 'true');
                $('#'.concat(i.toString(), '-', j.toString())).removeClass('locked')
            }
        }
    }
}

function getPuzzle() {
    var puzzle = Array.from(Array(9), _ => Array(9).fill(0));
    $('.sudoku-box').each(function() {
        if ($(this).hasClass('locked')){
            var index = $(this).attr('id').split("");
            puzzle[index[0]][index[2]] = parseInt($(this).html())
        }
    });
    return puzzle
}


// adjust sudoku grid on load
$(document).ready(function() {
    $('.modify-btn').data('mode', 'solve');
    adjustSudokuBox();
    generate("low")
});

// adjsut webpage when screen size changes
$( window ).resize(function() {
    adjustSudokuBox();
    if ($( window ).width() < 768){
        $('.navbar').each(function() {
            $(this).attr('data-toggle', "collapse");
            $(this).attr('data-target', "#navbarHeader");
        });
        $('.btn-div').each(function() {
            $(this).addClass('col-6');
            $(this).removeClass('col-12')
        })
    } else {
        $('.navbar').each(function() {
            $(this).removeAttr('data-toggle');
            $(this).removeAttr('data-target')
        });
        $('.btn-div').each(function() {
            $(this).addClass('col-12');
            $(this).removeClass('col-6')
        })
    }
});

// Prevent users from entering more than one digit per field
$('.sudoku-box').on('keydown paste', function(event) {
    var allowed_keys = [49, 50, 51, 52, 53, 54, 55, 56, 57];
    var always_allowed_keys = [8, 46];
    if(!(always_allowed_keys.includes(event.keyCode) ||($(this).text().length < 1 && allowed_keys.includes(event.keyCode)))) {
        event.preventDefault();
    } else {
        $(this).removeClass('error');
        if ($(".modify-btn" ).data('mode')=='modify') {
            if (always_allowed_keys.includes(event.keyCode)){
                $(this).removeClass('locked')
            } else {
                $(this).addClass('locked')
            }
        }
    }
});

//function to upload the image of a new sudoku
function upload(){
    $("#SudokuUpload").click();
}
$("#SudokuUpload").change(function () {
    $('.flash-box').fadeOut();
    $('#loading').find('h3').html('Loading Sudoku ...');
    $('#loading').css('display', 'flex');

    //remove the old sudoku
    $('.sudoku-box').each(function() {$(this).removeClass('locked')});
    $('.sudoku-box').each(function() {$(this).html("")});

    const file = $(this)[0].files[0];
    const reader = new FileReader();

    reader.onloadend = function () {
        alert(reader.result);
        $.ajax({
            type: "POST",
            url: "http://localhost:7071/api/predict",
            data: reader.result.split(',')[1],
            processData: false,
            contentType: false,
            success: readSuccess,
            error: errorOccurred
        });
    };

    const data = reader.readAsDataURL(file);

    function readSuccess(data) {
        data = JSON.parse(data);
        puzzle = data.puzzle;
        if (puzzle.length > 0){
            fillPuzzle(data.puzzle);
        }else{
            $('.flash-message').html('Sorry, no Sudoku was recognized in your picture (please upload an empty Sudoku).');
            $('.flash-box').fadeIn();
            setTimeout(function() {
                $('.flash-box').fadeOut()
            }, 10000)
        }
        setTimeout(function () {$('#loading').css('display', 'none')}, 250)
    }

});

//function to generate a newe Sudoku via a post request to the server
function generate(difficulty){
    $('.flash-box').fadeOut()
    $('#loading').find('h3').html('Generating Sudoku ...');
    $('#loading').css('display', 'flex');

    var dif_dict = {'difficulty': difficulty};

    $.ajax({
        type: "POST",
        url: "/generateSudoku",
        data: dif_dict,
        success: generateSuccess,
        error: errorOccurred,
        dataType: "json"
    });

    function generateSuccess(data){
        fillPuzzle(data.puzzle);
        $('.sudoku').data('solution', data.solution);
        setTimeout(function () {$('#loading').css('display', 'none')}, 250)
    }
}

//function to solve a sudoku by sending it to the server, b solution type it can be determined if the complete solution is displayed or only the errors are highlighted.
function solve(solutionType){
    $('.flash-box').fadeOut()
    if (solutionType == "check") {
        $('#loading').find('h3').html('Checking Sudoku ...');
    } else {
        $('#loading').find('h3').html('Solving Sudoku ...');
    }
    $('#loading').css('display', 'flex');

    var puzzle = getPuzzle();

    if (JSON.stringify(puzzle) == JSON.stringify($('.sudoku').data('puzzle')) && typeof $('.sudoku').data('solution') !== 'undefined'){
        solveSuccess({'solution' : $('.sudoku').data('solution'), 'valid': true})
    } else {
        $.ajax({
            type: "POST",
            url: "/solveSudoku",
            data: {'puzzle': JSON.stringify(puzzle)},
            success: solveSuccess,
            error: errorOccurred,
            dataType: "json"
        });
    }

    function solveSuccess(data) {
        var solution = data.solution;

        $('.sudoku').data('puzzle', puzzle);

        if(data.valid) {

            $('.sudoku').data('solution', solution);

            for (var i = 0; i < solution.length; i++) {
                for (var j = 0; j < solution[i].length; j++) {
                    if (puzzle[i][j] != 0) {
                        $('#'.concat(i.toString(), '-', j.toString())).html(puzzle[i][j].toString());
                        $('#'.concat(i.toString(), '-', j.toString())).attr('contenteditable', 'false');
                        $('#'.concat(i.toString(), '-', j.toString())).addClass('locked')
                    } else {
                        if (solutionType == "check") {
                            if (solution[i][j] != $('#'.concat(i.toString(), '-', j.toString())).html() && $('#'.concat(i.toString(), '-', j.toString())).html() != "") {
                                $('#'.concat(i.toString(), '-', j.toString())).addClass('error')
                            }
                        } else {
                            $('#'.concat(i.toString(), '-', j.toString())).html(solution[i][j].toString());
                        }
                        $('#'.concat(i.toString(), '-', j.toString())).attr('contenteditable', 'true');
                        $('#'.concat(i.toString(), '-', j.toString())).removeClass('locked')
                    }
                }
            }
        } else {
            $('.flash-message').html('Sorry, for your sudoku no solution was found.');
            $('.flash-box').fadeIn();
            setTimeout(function() {
                $('.flash-box').fadeOut()
            }, 10000)
        }
        //if (JSON.stringify(puzzle) == JSON.stringify($('.sudoku').data('puzzle')) && typeof $('.sudoku').data('solution') !== 'undefined') {
            setTimeout(function () {
                $('#loading').css('display', 'none')
            }, 500)
        //}
    }
}

//function to modify the sudoku to create a new one
$(".modify-btn" ).click(function() {
    if ($(this).data('mode')=='solve'){
        $(this).html("Finish");
        $(this).parent().siblings().each(function() { $(this).children('.btn').prop('disabled', true) });
        $(this).data('mode', 'modify');
        $('.sudoku-box').each(function() {
            $(this).attr('contenteditable', 'true');
        });
        if ($('.sudoku')[0].hasAttribute('data-solution')) {
            $('.sudoku').removeData('solution')
        }

    } else {
        $(this).html("Modify");
        $(this).parent().siblings().each(function() { $(this).children('.btn').prop('disabled', false) });
        $(this).data('mode', 'solve');
        $('.sudoku-box.locked').each(function() {
            $(this).attr('contenteditable', 'false');
        })
    }
});

function errorOccurred(){
    alert("Sorry an error occured while fetching the data. Please reload the page and try again")
}